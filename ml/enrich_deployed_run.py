"""One-off enrichment of an already-completed deployed-model MLflow run:
persists a SHAP background sample and a model-comparison summary as their
own artifacts (so /explain and /model/comparison at serving time don't need
the full data/ directory present), and backfills the deployed model's own
validation PR-AUC as a metric on that run (so serving's /model/metadata
endpoint has one single source of truth -- the run itself -- for everything
it reports)."""

import json
import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from ml.data import load_features_and_labels, prepare_model_matrix, time_based_split
from ml.evaluate import calibration_curve_data, evaluate_predictions
from ml.explain import global_shap_importance


def enrich_deployed_run(
    run_id: str,
    data_dir: str = "./data",
    mlflow_tracking_uri: str = "./mlruns",
    sample_size: int = 100,
    tmp_dir: str = "./ml/artifacts",
) -> dict:
    """Load deployed model, compute validation PR-AUC, and persist SHAP background.

    Parameters
    ----------
    run_id : str
        MLflow run ID of the already-logged deployed model.
    data_dir : str, default "./data"
        Path to data directory containing features.parquet and ground_truth.parquet.
    mlflow_tracking_uri : str, default "./mlruns"
        MLflow tracking URI (local filesystem or remote).
    sample_size : int, default 100
        Number of training samples to include in SHAP background.
    tmp_dir : str, default "./ml/artifacts"
        Temporary directory for writing CSV before logging as artifact.

    Returns
    -------
    dict
        Dictionary with keys: run_id, deployed_val_pr_auc, shap_background_path, model_comparison_path.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    features, labels = load_features_and_labels(data_dir)
    (train_X_raw, _), (val_X_raw, val_y), (test_X_raw, test_y) = time_based_split(
        features, labels
    )
    train_X = prepare_model_matrix(train_X_raw)
    val_X = prepare_model_matrix(val_X_raw)
    test_X = prepare_model_matrix(test_X_raw)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")
    raw_model = mlflow.sklearn.load_model(f"runs:/{run_id}/raw_deployed_model")
    val_pr_auc = evaluate_predictions(val_y, model.predict_proba(val_X)[:, 1])["pr_auc"]

    sample = train_X.sample(min(sample_size, len(train_X)), random_state=42)
    os.makedirs(tmp_dir, exist_ok=True)
    bg_path = f"{tmp_dir}/shap_background.csv"
    sample.to_csv(bg_path, index=False)

    test_scores = model.predict_proba(test_X)[:, 1]
    prob_true, prob_pred = calibration_curve_data(test_y, test_scores)
    shap_sample = test_X.sample(min(2000, len(test_X)), random_state=42)
    shap_importance = global_shap_importance(raw_model, shap_sample, background=sample)

    comparison = {
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
        "shap_importances": {k: float(v) for k, v in shap_importance.items()},
    }
    comparison_path = f"{tmp_dir}/model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f)

    client = MlflowClient()
    client.log_artifact(run_id, bg_path)
    client.log_artifact(run_id, comparison_path)
    client.log_metric(run_id, "deployed_val_pr_auc", val_pr_auc)

    return {
        "run_id": run_id,
        "deployed_val_pr_auc": val_pr_auc,
        "shap_background_path": bg_path,
        "model_comparison_path": comparison_path,
    }
