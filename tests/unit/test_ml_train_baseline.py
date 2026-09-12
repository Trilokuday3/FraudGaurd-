import numpy as np
import pandas as pd

from ml.train import train_baseline, train_random_forest


def _tiny_fixture():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + 0.5 * X["b"] > 0).astype(int)
    return X, y


def test_train_baseline_fits_and_predicts():
    X, y = _tiny_fixture()
    model = train_baseline(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_train_random_forest_fits_and_predicts():
    X, y = _tiny_fixture()
    model = train_random_forest(X, y, n_estimators=20)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert (probs >= 0).all() and (probs <= 1).all()
