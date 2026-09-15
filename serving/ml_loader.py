"""Loads the deployed model, Isolation Forest, and SHAP background sample
from one config-named MLflow run, once, at app startup.

Loads two versions of the deployed model: the calibrated wrapper
(CalibratedClassifierCV) for actual scoring, and the raw pre-calibration
estimator for SHAP -- shap.TreeExplainer/LinearExplainer both need direct
access to a real linear or tree model's internals, which a calibration
wrapper does not expose (confirmed via shap.InvalidModelError when tried
directly against the wrapper)."""

import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


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
