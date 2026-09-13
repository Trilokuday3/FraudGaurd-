import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.explain import explain_prediction, global_shap_importance


def _tiny_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=2, random_state=42)
    model.fit(X, y)
    return model, X


def test_global_shap_importance_ranks_informative_feature_first():
    model, X = _tiny_model()
    importance = global_shap_importance(model, X)
    assert next(iter(importance.keys())) == "a"


def test_global_shap_importance_is_sorted_descending():
    model, X = _tiny_model()
    importance = global_shap_importance(model, X)
    values = list(importance.values())
    assert values == sorted(values, reverse=True)


def test_global_shap_importance_covers_all_features():
    model, X = _tiny_model()
    importance = global_shap_importance(model, X)
    assert set(importance.keys()) == {"a", "b", "c"}


def test_explain_prediction_returns_per_feature_contributions():
    model, X = _tiny_model()
    row = X.iloc[[0]]
    contributions = explain_prediction(model, row)
    assert set(contributions.keys()) == {"a", "b", "c"}
    assert all(isinstance(v, (int, float, np.floating)) for v in contributions.values())


def _tiny_linear_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, X


def test_global_shap_importance_works_for_linear_model():
    model, X = _tiny_linear_model()
    importance = global_shap_importance(model, X, background=X)
    assert next(iter(importance.keys())) == "a"
    assert set(importance.keys()) == {"a", "b", "c"}


def test_explain_prediction_linear_model_is_not_degenerate_with_background():
    # A single row cannot be its own SHAP background (the baseline
    # expectation collapses to that exact row, zeroing every contribution)
    # -- this test guards against exactly that regression.
    model, X = _tiny_linear_model()
    row = X.iloc[[0]]
    contributions = explain_prediction(model, row, background=X)
    assert any(abs(v) > 1e-6 for v in contributions.values())
