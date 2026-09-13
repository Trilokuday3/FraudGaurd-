"""CLI: python -m ml train [--data-dir DIR]"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import mlflow

from ml.calibration import calibrate
from ml.data import load_features_and_labels, prepare_model_matrix, time_based_split
from ml.evaluate import calibration_curve_data, evaluate_predictions
from ml.explain import explain_prediction, global_shap_importance
from ml.train import (
    isolation_forest_anomaly_scores,
    select_champion,
    select_deployed_model,
    train_baseline,
    train_isolation_forest,
    train_lightgbm,
    train_random_forest,
    train_xgboost,
)

ARTIFACTS_DIR = "./ml/artifacts"


def _run_training(data_dir: str = "./data", mlflow_tracking_uri: str = "./mlruns") -> dict:
    """Run the full modeling pipeline end to end.

    Loads features and labels, time-splits them, trains the baseline,
    Random Forest, XGBoost, and LightGBM models, selects the overall
    best-performing candidate by validation PR-AUC (including the
    baseline -- not just the boosted-tree champion), calibrates it,
    evaluates it on the test set, trains an Isolation Forest anomaly
    detector, computes global and local SHAP explanations, logs
    everything to MLflow, and writes ``results.json`` plus
    calibration-curve and SHAP-importance plot artifacts to
    ``ARTIFACTS_DIR``.

    Parameters
    ----------
    data_dir : str, default="./data"
        Directory containing ``features.parquet`` and
        ``ground_truth.parquet``.
    mlflow_tracking_uri : str, default="./mlruns"
        MLflow tracking URI to log runs to.

    Returns
    -------
    dict
        Results summary: deployed model name, boosted-tree champion name,
        validation/test metrics for every model, Isolation Forest anomaly
        summary, global SHAP importance, a local SHAP example,
        calibration curve data, and row counts.
    """
    features, labels = load_features_and_labels(data_dir)
    (train_X_raw, train_y), (val_X_raw, val_y), (test_X_raw, test_y) = time_based_split(
        features, labels
    )
    train_X = prepare_model_matrix(train_X_raw)
    val_X = prepare_model_matrix(val_X_raw)
    test_X = prepare_model_matrix(test_X_raw)

    # mlflow>=3.x refuses the filesystem tracking backend (e.g. "./mlruns")
    # unless explicitly opted into via this env var -- see
    # https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store.
    # Filesystem tracking is the design this project uses (mlruns/ is
    # already gitignored from sub-project 1), so opt in rather than
    # migrating to a database backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fraudguard-modeling")

    with mlflow.start_run(run_name="baseline"):
        baseline = train_baseline(train_X, train_y)
        baseline_val_metrics = evaluate_predictions(val_y, baseline.predict_proba(val_X)[:, 1])
        mlflow.log_metric("val_pr_auc", baseline_val_metrics["pr_auc"])

    with mlflow.start_run(run_name="random_forest"):
        rf = train_random_forest(train_X, train_y)
        rf_val_metrics = evaluate_predictions(val_y, rf.predict_proba(val_X)[:, 1])
        mlflow.log_metric("val_pr_auc", rf_val_metrics["pr_auc"])

    with mlflow.start_run(run_name="xgboost"):
        xgb_model = train_xgboost(train_X, train_y)
        xgb_val_metrics = evaluate_predictions(val_y, xgb_model.predict_proba(val_X)[:, 1])
        mlflow.log_metric("val_pr_auc", xgb_val_metrics["pr_auc"])

    with mlflow.start_run(run_name="lightgbm"):
        lgbm_model = train_lightgbm(train_X, train_y)
        lgbm_val_metrics = evaluate_predictions(val_y, lgbm_model.predict_proba(val_X)[:, 1])
        mlflow.log_metric("val_pr_auc", lgbm_val_metrics["pr_auc"])

    # champion_model itself is unused from here on: Task 9 only needs
    # champion_name (kept as an informational field) since the model
    # that actually gets calibrated/evaluated/explained is deployed_model
    # below (the overall best of all four candidates, not just the
    # boosted-tree comparison).
    champion_name, _champion_model = select_champion(
        xgb_model, xgb_val_metrics, lgbm_model, lgbm_val_metrics
    )

    candidates = {
        "baseline": (baseline, baseline_val_metrics),
        "random_forest": (rf, rf_val_metrics),
        "xgboost": (xgb_model, xgb_val_metrics),
        "lightgbm": (lgbm_model, lgbm_val_metrics),
    }
    deployed_name, deployed_model = select_deployed_model(candidates)

    calibrated_deployed = calibrate(deployed_model, val_X, val_y, method="isotonic")

    test_scores = calibrated_deployed.predict_proba(test_X)[:, 1]
    test_metrics = evaluate_predictions(test_y, test_scores)

    # Compare against a CALIBRATED baseline, not the baseline's raw scores:
    # comparing a calibrated deployed model to an uncalibrated baseline can
    # make an identical model look worse than itself purely from
    # calibration's tie-collapsing effect on a rank-based metric like
    # PR-AUC (isotonic regression fit on validation can map several
    # distinct test scores to the same step). When the baseline IS the
    # deployed model, reuse its calibration instead of recomputing it, so
    # the two numbers are mathematically identical, not just close.
    calibrated_baseline = (
        calibrated_deployed
        if deployed_name == "baseline"
        else calibrate(baseline, val_X, val_y, method="isotonic")
    )
    baseline_test_metrics = evaluate_predictions(
        test_y, calibrated_baseline.predict_proba(test_X)[:, 1]
    )

    isolation_forest = train_isolation_forest(train_X)
    anomaly_scores = isolation_forest_anomaly_scores(isolation_forest, test_X)

    sample_size = min(2000, len(test_X))
    shap_sample = test_X.sample(sample_size, random_state=42)
    shap_background = train_X.sample(min(100, len(train_X)), random_state=42)
    shap_importance = global_shap_importance(
        deployed_model, shap_sample, background=shap_background
    )
    shap_local_example = explain_prediction(
        deployed_model, test_X.iloc[[0]], background=shap_background
    )

    prob_true, prob_pred = calibration_curve_data(test_y, test_scores)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Calibration curve plot (spec: "saved as a plot artifact")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="deployed model (calibrated)")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.legend()
    fig.savefig(f"{ARTIFACTS_DIR}/calibration_curve.png")
    plt.close(fig)

    # Global SHAP importance bar chart (spec: "saved as a bar-chart artifact")
    top_features = list(shap_importance.items())[:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [name for name, _ in reversed(top_features)],
        [val for _, val in reversed(top_features)],
    )
    ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(f"{ARTIFACTS_DIR}/shap_global_importance.png")
    plt.close(fig)

    results = {
        "deployed_model_name": deployed_name,
        "champion_name": champion_name,
        "baseline_val_metrics": baseline_val_metrics,
        "random_forest_val_metrics": rf_val_metrics,
        "xgboost_val_metrics": xgb_val_metrics,
        "lightgbm_val_metrics": lgbm_val_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "isolation_forest_test_anomaly_mean": float(anomaly_scores.mean()),
        "shap_global_importance": {k: float(v) for k, v in shap_importance.items()},
        "shap_local_example": {
            "transaction_id": str(test_X_raw["transaction_id"].iloc[0]),
            "contributions": {k: float(v) for k, v in shap_local_example.items()},
        },
        "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
        "row_counts": {"train": len(train_X), "val": len(val_X), "test": len(test_X)},
    }

    with mlflow.start_run(run_name="deployed_model_final"):
        mlflow.log_param("deployed_model", deployed_name)
        mlflow.log_param("boosted_tree_champion", champion_name)
        mlflow.log_metric("test_pr_auc", test_metrics["pr_auc"])
        mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])
        mlflow.log_artifact(f"{ARTIFACTS_DIR}/calibration_curve.png")
        mlflow.log_artifact(f"{ARTIFACTS_DIR}/shap_global_importance.png")

    with open(f"{ARTIFACTS_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml")
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="Train baseline through champion + Isolation Forest")
    train_cmd.add_argument("--data-dir", type=str, default="./data")

    args = parser.parse_args()

    if args.command == "train":
        results = _run_training(data_dir=args.data_dir)
        print(f"Boosted-tree champion (XGBoost vs LightGBM): {results['champion_name']}")
        print(f"Deployed model (best of all candidates):     {results['deployed_model_name']}")
        print(f"Baseline test PR-AUC:       {results['baseline_test_metrics']['pr_auc']:.4f}")
        print(f"Deployed model test PR-AUC: {results['test_metrics']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
