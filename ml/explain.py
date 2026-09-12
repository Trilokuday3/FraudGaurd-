"""SHAP explainability: global feature importance + per-prediction local
explanations. Uses TreeExplainer (fast, exact for tree ensembles) rather
than KernelExplainer (model-agnostic but slow) since the champion is always
XGBoost or LightGBM.

Note on SHAP return shapes (verified against shap==0.52.0 installed in this
project's venv): for a binary XGBClassifier/LGBMClassifier,
``TreeExplainer(model).shap_values(X)`` returns a plain 2D ``np.ndarray`` of
shape ``(n_samples, n_features)`` -- the contribution to the positive
class's raw margin -- not a ``list`` of one array per class as older SHAP
versions returned. ``_tree_shap_values`` below still handles the older
list-per-class shape and a 3D ``(n_samples, n_features, n_classes)`` shape
(seen in some newer/multiclass configurations) defensively, but the 2D
array is what actually comes back in this environment.
"""

import numpy as np
import pandas as pd
import shap


def _tree_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for a tree-based model, normalized to 2D.

    Handles the SHAP API's return-type drift across versions: older
    versions return a ``list`` of one array per class for classifiers,
    some versions return a 3D array of shape
    ``(n_samples, n_features, n_classes)``, and the version installed in
    this project (0.52.0) returns a plain 2D array of shape
    ``(n_samples, n_features)`` directly for binary classifiers. In every
    case this function returns the positive-class contributions as a 2D
    array of shape ``(n_samples, n_features)``.

    Parameters
    ----------
    model : object
        Fitted XGBoost or LightGBM classifier.
    X : pd.DataFrame
        Feature matrix to explain.

    Returns
    -------
    np.ndarray
        SHAP values of shape ``(n_samples, n_features)``.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[:, :, -1]
    return values


def global_shap_importance(model, X_sample: pd.DataFrame) -> dict:
    """Compute global feature importance as mean absolute SHAP value.

    Parameters
    ----------
    model : object
        Fitted XGBoost or LightGBM classifier.
    X_sample : pd.DataFrame
        Sample of feature rows to compute importance over.

    Returns
    -------
    dict
        Feature name -> mean absolute SHAP value, sorted descending by
        importance.
    """
    values = _tree_shap_values(model, X_sample)
    mean_abs = np.abs(values).mean(axis=0)
    return dict(sorted(zip(X_sample.columns, mean_abs), key=lambda kv: kv[1], reverse=True))


def explain_prediction(model, feature_row: pd.DataFrame) -> dict:
    """Compute per-feature SHAP contributions for a single prediction.

    This is the function sub-project 4's ``/explain`` API endpoint calls
    directly, so its return shape (a flat feature-name -> contribution
    dict) is a stable contract for that future consumer.

    Parameters
    ----------
    model : object
        Fitted XGBoost or LightGBM classifier.
    feature_row : pd.DataFrame
        Single-row feature matrix to explain.

    Returns
    -------
    dict
        Feature name -> SHAP contribution for this row.
    """
    values = _tree_shap_values(model, feature_row)
    return dict(zip(feature_row.columns, values[0]))
