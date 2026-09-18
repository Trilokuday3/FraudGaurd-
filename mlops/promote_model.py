"""Promote an MLflow run to the registered model's "champion" alias, gated
on beating the current champion's validation PR-AUC, then re-sync the
serving layer's config (decision/thresholds.json, deploy/model_store/) to
the newly-promoted run -- the same three steps infra/deploy.md's manual
"re-baking after a model retrain" section already documents, now automated
and fronted by a real promotion gate.

The registry (not MLFLOW_RUN_ID/thresholds.json) is the source of truth for
which run *should* be deployed; thresholds.json and deploy/model_store/
stay as the serving layer's actual read path, unchanged from sub-project 8,
kept in sync here rather than replaced."""

import argparse
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from decision.select_thresholds import select_thresholds_for_run
from ml.enrich_deployed_run import enrich_deployed_run
from scripts.vendor_model_store import vendor_model_store

REGISTERED_MODEL_NAME = "fraudguard-fraud-model"


def _get_alias_version(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def promote_model(
    run_id: str,
    mlflow_tracking_uri: str = "./mlruns",
    data_dir: str = "./data",
    model_store_dir: str = "./deploy/model_store",
    thresholds_path: str = "./decision/thresholds.json",
    registered_model_name: str = REGISTERED_MODEL_NAME,
    force: bool = False,
) -> dict:
    """Promote `run_id` to the "champion" alias of `registered_model_name`.

    Refuses (raises ValueError) if a current champion exists and
    `run_id`'s validation PR-AUC doesn't at least match it, unless
    `force=True`. On success, re-points "previous" at the outgoing
    champion (so `rollback_model.rollback_model` can undo this exact
    promotion) and re-syncs thresholds.json / deploy/model_store/ to the
    new champion.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # enrich_deployed_run is idempotent (re-logging the same artifact/metric
    # is safe -- see its own docstring) and is what actually computes
    # deployed_val_pr_auc, so run it unconditionally rather than assuming
    # a prior manual run already did.
    enrich_result = enrich_deployed_run(
        run_id, data_dir=data_dir, mlflow_tracking_uri=mlflow_tracking_uri
    )
    candidate_val_pr_auc = enrich_result["deployed_val_pr_auc"]

    current_champion = _get_alias_version(client, registered_model_name, "champion")
    if current_champion is not None and not force:
        champion_run = client.get_run(current_champion.run_id)
        champion_val_pr_auc = champion_run.data.metrics.get("deployed_val_pr_auc", 0.0)
        if candidate_val_pr_auc < champion_val_pr_auc:
            raise ValueError(
                f"Candidate run {run_id} val_pr_auc={candidate_val_pr_auc:.4f} does not "
                f"beat current champion val_pr_auc={champion_val_pr_auc:.4f}. "
                "Pass force=True to promote anyway."
            )

    new_version = mlflow.register_model(
        f"runs:/{run_id}/calibrated_deployed_model", registered_model_name
    )

    previous_champion_run_id = None
    if current_champion is not None and current_champion.run_id != run_id:
        client.set_registered_model_alias(
            registered_model_name, "previous", current_champion.version
        )
        previous_champion_run_id = current_champion.run_id

    client.set_registered_model_alias(registered_model_name, "champion", new_version.version)

    select_thresholds_for_run(
        run_id,
        data_dir=data_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        output_path=thresholds_path,
    )
    vendor_model_store(run_id, mlflow_tracking_uri, model_store_dir)

    return {
        "promoted_run_id": run_id,
        "champion_version": new_version.version,
        "previous_champion_run_id": previous_champion_run_id,
        "candidate_val_pr_auc": candidate_val_pr_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="promote_model")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force", action="store_true", help="Skip the PR-AUC comparison gate")
    args = parser.parse_args()

    result = promote_model(args.run_id, force=args.force)
    print(
        f"Promoted {result['promoted_run_id']} to champion "
        f"(registry version {result['champion_version']}, "
        f"val_pr_auc={result['candidate_val_pr_auc']:.4f})"
    )


if __name__ == "__main__":
    main()
