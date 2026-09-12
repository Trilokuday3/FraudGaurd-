import numpy as np
import pandas as pd

from ml.train import select_champion, train_lightgbm, train_xgboost


def _tiny_fixture():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + 0.5 * X["b"] > 0).astype(int)
    return X, y


def test_train_xgboost_fits_and_predicts():
    X, y = _tiny_fixture()
    model = train_xgboost(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_train_lightgbm_fits_and_predicts():
    X, y = _tiny_fixture()
    model = train_lightgbm(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_select_champion_picks_higher_pr_auc():
    name, model = select_champion(
        xgb_model="XGB_MODEL",
        xgb_val_metrics={"pr_auc": 0.42},
        lgbm_model="LGBM_MODEL",
        lgbm_val_metrics={"pr_auc": 0.55},
    )
    assert name == "lightgbm"
    assert model == "LGBM_MODEL"


def test_select_champion_ties_prefer_xgboost():
    name, model = select_champion(
        xgb_model="XGB_MODEL",
        xgb_val_metrics={"pr_auc": 0.5},
        lgbm_model="LGBM_MODEL",
        lgbm_val_metrics={"pr_auc": 0.5},
    )
    assert name == "xgboost"
    assert model == "XGB_MODEL"
