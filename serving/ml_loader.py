"""Loads the deployed model, Isolation Forest, and SHAP background sample
from one config-named MLflow run, once, at app startup.

Loads two versions of the deployed model: the calibrated wrapper
(CalibratedClassifierCV) for actual scoring, and the raw pre-calibration
estimator for SHAP -- shap.TreeExplainer/LinearExplainer both need direct
access to a real linear or tree model's internals, which a calibration
wrapper does not expose (confirmed via shap.InvalidModelError when tried
directly against the wrapper)."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


def _rewrite_meta_field(meta_path, field: str, correct_value: str) -> None:
    content = meta_path.read_text()
    repaired = re.sub(
        rf"^{field}:.*$",
        f"{field}: {correct_value}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if repaired != content:
        meta_path.write_text(repaired)


def _repair_artifact_uris(tracking_uri: str) -> None:
    """Rewrite every run's and every logged model's absolute artifact path
    in this file-store tree to point at `tracking_uri`'s actual current
    location on disk.

    MLflow's local file store bakes an absolute path into each run's
    `meta.yaml` (`artifact_uri`) AND, separately, into each "Logged Model"
    entity's own `meta.yaml` under `<experiment_id>/models/m-<hash>/`
    (`artifact_location` -- a different field name, a different directory).
    `mlflow.sklearn.log_model` routes actual model files (MLmodel/model.pkl)
    through the models/ registry, not the run's own artifacts/ folder, so
    both paths need repairing -- a run-only fix left
    `mlflow.sklearn.load_model("runs:/<id>/<name>")` still raising "Failed
    to download artifacts ... please ensure the path is correct" against a
    freshly vendored copy, confirmed directly, until this also covered
    models/*/meta.yaml.

    A vendored/relocated copy of an `mlruns/`-style tree (e.g.
    `deploy/model_store/`, copied from a dev machine and deployed to a
    different host with a different absolute path) still has the
    *original* machine's paths baked in everywhere until this runs. Safe to
    call on a normal, never-moved local `mlruns/` too: it rewrites each
    path to the same value it already has, so it's a no-op there other than
    touching file mtimes.
    """
    root = Path(tracking_uri).resolve()
    if not root.is_dir():
        return
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if run_dir.name == "models" or not run_dir.is_dir():
                continue
            meta_path = run_dir / "meta.yaml"
            if meta_path.exists():
                _rewrite_meta_field(
                    meta_path, "artifact_uri", f"file:{(run_dir / 'artifacts').as_posix()}"
                )

        models_dir = exp_dir / "models"
        if not models_dir.is_dir():
            continue
        for model_dir in models_dir.iterdir():
            meta_path = model_dir / "meta.yaml"
            if not model_dir.is_dir() or not meta_path.exists():
                continue
            _rewrite_meta_field(
                meta_path,
                "artifact_location",
                f"file:{(model_dir / 'artifacts').as_posix()}",
            )


@dataclass
class LoadedModel:
    """Container for a loaded deployed model and its supporting artifacts.

    Attributes
    ----------
    model : object
        The fitted sklearn model (typically CalibratedClassifierCV).
    raw_model : object
        The pre-calibration estimator, used only for SHAP.
    isolation_forest : object
        The fitted IsolationForest anomaly detector.
    shap_background : pd.DataFrame
        Background sample for SHAP model explanations.
    run_id : str
        The MLflow run ID from which these artifacts were loaded.
    """

    model: object
    raw_model: object
    isolation_forest: object
    shap_background: pd.DataFrame
    run_id: str


def load_deployed_model(run_id: str, mlflow_tracking_uri: str = "./mlruns") -> LoadedModel:
    """Load the deployed model, Isolation Forest, and SHAP background from MLflow.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the deployed model, Isolation Forest, and
        SHAP background artifacts.
    mlflow_tracking_uri : str, default "./mlruns"
        MLflow tracking URI (local filesystem or remote).

    Returns
    -------
    LoadedModel
        Dataclass containing model, raw_model, isolation_forest,
        shap_background, and run_id.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    _repair_artifact_uris(mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")
    raw_model = mlflow.sklearn.load_model(f"runs:/{run_id}/raw_deployed_model")
    isolation_forest = mlflow.sklearn.load_model(f"runs:/{run_id}/isolation_forest_model")

    client = MlflowClient()
    local_path = client.download_artifacts(run_id, "shap_background.csv")
    shap_background = pd.read_csv(local_path)

    return LoadedModel(
        model=model,
        raw_model=raw_model,
        isolation_forest=isolation_forest,
        shap_background=shap_background,
        run_id=run_id,
    )
