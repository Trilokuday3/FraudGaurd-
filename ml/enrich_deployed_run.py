"""One-off enrichment of an already-completed deployed-model MLflow run:
persists a SHAP background sample as its own artifact (so /explain at
serving time doesn't need the full data/ directory present), and backfills
the deployed model's own validation PR-AUC as a metric on that run (so
serving's /model/metadata endpoint has one single source of truth -- the
run itself -- for everything it reports)."""

import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from ml.data import load_features_and_labels, prepare_model_matrix, time_based_split
from ml.evaluate import evaluate_predictions


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
        Dictionary with keys: run_id, deployed_val_pr_auc, shap_background_path.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    features, labels = load_features_and_labels(data_dir)
    (train_X_raw, _), (val_X_raw, val_y), _ = time_based_split(features, labels)
    train_X = prepare_model_matrix(train_X_raw)
    val_X = prepare_model_matrix(val_X_raw)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")
    val_pr_auc = evaluate_predictions(val_y, model.predict_proba(val_X)[:, 1])["pr_auc"]

    sample = train_X.sample(min(sample_size, len(train_X)), random_state=42)
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = f"{tmp_dir}/shap_background.csv"
    sample.to_csv(local_path, index=False)

    client = MlflowClient()
    client.log_artifact(run_id, local_path)
    client.log_metric(run_id, "deployed_val_pr_auc", val_pr_auc)

    return {
        "run_id": run_id,
        "deployed_val_pr_auc": val_pr_auc,
        "shap_background_path": local_path,
    }
