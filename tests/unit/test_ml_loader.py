import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from serving.ml_loader import load_deployed_model


def test_load_deployed_model_round_trips(tmp_path):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_dir = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("test-loader")

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    iso = IsolationForest(random_state=42).fit(X)
    background = X.sample(10, random_state=42)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(
            model, name="calibrated_deployed_model", serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            iso, name="isolation_forest_model", serialization_format="cloudpickle"
        )
        bg_path = tmp_path / "shap_background.csv"
        background.to_csv(bg_path, index=False)
        mlflow.log_artifact(str(bg_path))

    loaded = load_deployed_model(run_id, mlflow_tracking_uri=mlflow_dir)

    assert loaded.run_id == run_id
    assert loaded.model.predict_proba(X)[:, 1].shape == (50,)
    assert loaded.isolation_forest.score_samples(X).shape == (50,)
    assert len(loaded.shap_background) == 10
