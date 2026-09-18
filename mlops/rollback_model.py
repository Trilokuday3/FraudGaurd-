"""Revert the registered model's "champion" alias to whatever "previous"
currently points at, re-syncing the serving layer's config the same way
promote_model does. Swaps the two aliases rather than just jumping to
"previous" and discarding it, so calling this twice in a row returns to
the state before the first call -- a rollback of a rollback is itself a
(re-)promotion, not a dead end."""

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient

from decision.select_thresholds import select_thresholds_for_run
from mlops.promote_model import REGISTERED_MODEL_NAME, _get_alias_version
from scripts.vendor_model_store import vendor_model_store


def rollback_model(
    mlflow_tracking_uri: str = "./mlruns",
    data_dir: str = "./data",
    model_store_dir: str = "./deploy/model_store",
    thresholds_path: str = "./decision/thresholds.json",
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> dict:
    """Swap "champion" and "previous" so the registry's previous champion
    becomes champion again, then re-sync thresholds.json / deploy/model_store/
    to that run. Raises ValueError if no "previous" alias is set (nothing
    to roll back to -- e.g. only one promotion has ever happened)."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    previous = _get_alias_version(client, registered_model_name, "previous")
    if previous is None:
        raise ValueError(
            f"No 'previous' alias set for '{registered_model_name}'; " "nothing to roll back to."
        )
    champion = _get_alias_version(client, registered_model_name, "champion")

    client.set_registered_model_alias(registered_model_name, "previous", champion.version)
    client.set_registered_model_alias(registered_model_name, "champion", previous.version)

    select_thresholds_for_run(
        previous.run_id,
        data_dir=data_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        output_path=thresholds_path,
    )
    vendor_model_store(previous.run_id, mlflow_tracking_uri, model_store_dir)

    return {
        "rolled_back_to_run_id": previous.run_id,
        "champion_version": previous.version,
        "previous_champion_run_id": champion.run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="rollback_model")
    parser.parse_args()

    result = rollback_model()
    print(
        f"Rolled back to {result['rolled_back_to_run_id']} "
        f"(registry version {result['champion_version']})"
    )


if __name__ == "__main__":
    main()
