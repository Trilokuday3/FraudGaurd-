"""SHAP explainability: global feature importance + per-prediction local
explanations. Uses TreeExplainer (fast, exact) for tree ensembles
(XGBoost/LightGBM/RandomForest) and LinearExplainer for linear models
(e.g. the logistic-regression baseline, when it is the actual
best-performing candidate) -- never KernelExplainer (model-agnostic but
slow), since every candidate this project trains is one or the other.

Note on SHAP return shapes (verified against shap==0.52.0 installed in this
project's venv): for a binary XGBClassifier/LGBMClassifier,
``TreeExplainer(model).shap_values(X)`` returns a plain 2D ``np.ndarray`` of
shape ``(n_samples, n_features)`` directly -- not a ``list`` of one array
per class as older SHAP versions returned. ``_model_shap_values`` below
still handles the older list-per-class shape and a 3D
``(n_samples, n_features, n_classes)`` shape defensively, but the 2D array
is what actually comes back in this environment for both explainer types.
"""

import numpy as np
import pandas as pd
import shap

_LINEAR_BACKGROUND_SIZE = 100


def _model_shap_values(model, X: pd.DataFrame, background: pd.DataFrame | None) -> np.ndarray:
    """Compute SHAP values for either a tree-based or linear model, normalized to 2D.

    Parameters
    ----------
    model : object
        Fitted classifier -- a tree ensemble (XGBoost/LightGBM/RandomForest)
        or a linear model (e.g. LogisticRegression, detected via `coef_`).
    X : pd.DataFrame
        Feature rows to explain.
    background : pd.DataFrame or None
        Reference sample for LinearExplainer's expected-value baseline.
        Ignored for tree models. A single row cannot meaningfully be its
        own background (the baseline expectation collapses to that exact
        row, zeroing every contribution), so a real reference sample
        matters when explaining one row of a linear model.

    Returns
    -------
    np.ndarray
        SHAP values of shape ``(n_samples, n_features)``.
    """
    if hasattr(model, "coef_"):
        reference = background if background is not None else X
        sample_size = min(_LINEAR_BACKGROUND_SIZE, len(reference))
        reference = (
            reference.sample(sample_size, random_state=42)
            if len(reference) > sample_size
            else reference
        )
        explainer = shap.LinearExplainer(model, reference)
    else:
        explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[:, :, -1]
    return values


def global_shap_importance(
    model, X_sample: pd.DataFrame, background: pd.DataFrame | None = None
) -> dict:
    """Compute global feature importance as mean absolute SHAP value.

    Parameters
    ----------
    model : object
        Fitted classifier (tree ensemble or linear model).
    X_sample : pd.DataFrame
        Sample of feature rows to compute importance over.
    background : pd.DataFrame, optional
        Reference sample for a linear model's SHAP baseline. Ignored for
        tree models. Defaults to `X_sample` itself if omitted.

    Returns
    -------
    dict
        Feature name -> mean absolute SHAP value, sorted descending by
        importance.
    """
    values = _model_shap_values(model, X_sample, background)
    mean_abs = np.abs(values).mean(axis=0)
    return dict(sorted(zip(X_sample.columns, mean_abs), key=lambda kv: kv[1], reverse=True))


def explain_prediction(
    model, feature_row: pd.DataFrame, background: pd.DataFrame | None = None
) -> dict:
    """Compute per-feature SHAP contributions for a single prediction.

    This is the function sub-project 4's ``/explain`` API endpoint calls
    directly, so its return shape (a flat feature-name -> contribution
    dict) is a stable contract for that future consumer.

    Parameters
    ----------
    model : object
        Fitted classifier (tree ensemble or linear model).
    feature_row : pd.DataFrame
        Single-row feature matrix to explain.
    background : pd.DataFrame, optional
        Reference sample for a linear model's SHAP baseline -- required
        for a non-degenerate explanation when `model` is linear, since a
        single row cannot be its own baseline. Ignored for tree models.

    Returns
    -------
    dict
        Feature name -> SHAP contribution for this row.
    """
    values = _model_shap_values(model, feature_row, background)
    return dict(zip(feature_row.columns, values[0]))
