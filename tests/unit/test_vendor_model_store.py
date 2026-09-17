"""Tests for scripts/vendor_model_store.py."""

import os

import mlflow

from scripts.vendor_model_store import vendor_model_store


def test_vendor_model_store_copies_deployed_and_candidate_runs(tmp_path):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    tracking_uri = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("fraudguard-modeling")

    with mlflow.start_run(experiment_id=experiment_id, run_name="baseline") as run:
        mlflow.log_metric("val_pr_auc", 0.5)
        deployed_run_id = run.info.run_id

    with mlflow.start_run(experiment_id=experiment_id, run_name="xgboost") as run:
        mlflow.log_metric("val_pr_auc", 0.6)

    output_dir = str(tmp_path / "model_store")
    run_ids = vendor_model_store(deployed_run_id, tracking_uri, output_dir)

    assert run_ids[0] == deployed_run_id
    assert len(run_ids) == 2  # deployed (baseline) + xgboost candidate found

    # Repoint a fresh client at the vendored copy and confirm it loads.
    mlflow.set_tracking_uri(output_dir)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    vendored_run = client.get_run(deployed_run_id)
    assert vendored_run.data.metrics["val_pr_auc"] == 0.5
