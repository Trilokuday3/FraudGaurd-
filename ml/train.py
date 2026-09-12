"""Model training: baseline -> Random Forest -> XGBoost/LightGBM champion ->
Isolation Forest. Every classifier here uses class-weighting for the ~1.5%
imbalance, never resampling -- see the spec's "Why time-based split" and
"Class imbalance handling" sections for the reasoning."""

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_baseline(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression baseline model with balanced class weighting.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or array-like
        Training labels.

    Returns
    -------
    LogisticRegression
        Fitted baseline model.
    """
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200) -> RandomForestClassifier:
    """Train a random forest model with balanced class weighting.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or array-like
        Training labels.
    n_estimators : int, default=200
        Number of trees in the forest.

    Returns
    -------
    RandomForestClassifier
        Fitted random forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _scale_pos_weight(y_train) -> float:
    """Compute the negative/positive class count ratio from the training fold.

    This is the XGBoost/LightGBM analog of ``class_weight="balanced"`` --
    neither library exposes that exact parameter, so the ratio is passed
    explicitly via ``scale_pos_weight``. Must always be computed from the
    training fold only, never validation or test, to avoid leakage.

    Parameters
    ----------
    y_train : pd.Series or array-like
        Training labels.

    Returns
    -------
    float
        Ratio of negative to positive samples in ``y_train``.
    """
    positive = (y_train == 1).sum()
    negative = (y_train == 0).sum()
    return negative / max(positive, 1)


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with scale_pos_weight for class imbalance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or array-like
        Training labels.

    Returns
    -------
    xgb.XGBClassifier
        Fitted XGBoost model.
    """
    model = xgb.XGBClassifier(
        scale_pos_weight=_scale_pos_weight(y_train),
        n_estimators=200,
        max_depth=5,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with scale_pos_weight for class imbalance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or array-like
        Training labels.

    Returns
    -------
    lgb.LGBMClassifier
        Fitted LightGBM model.
    """
    model = lgb.LGBMClassifier(
        scale_pos_weight=_scale_pos_weight(y_train),
        n_estimators=200,
        max_depth=5,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def select_champion(
    xgb_model: object,
    xgb_val_metrics: dict,
    lgbm_model: object,
    lgbm_val_metrics: dict,
) -> tuple[str, object]:
    """Pick the champion model by validation PR-AUC.

    Pure comparison: no training, no I/O, no randomness. Ties go to
    ``"xgboost"`` by design.

    Parameters
    ----------
    xgb_model : object
        Fitted XGBoost model.
    xgb_val_metrics : dict
        Validation metrics for ``xgb_model``, as returned by
        ``ml.evaluate.evaluate_predictions`` (must contain ``"pr_auc"``).
    lgbm_model : object
        Fitted LightGBM model.
    lgbm_val_metrics : dict
        Validation metrics for ``lgbm_model``, as returned by
        ``ml.evaluate.evaluate_predictions`` (must contain ``"pr_auc"``).

    Returns
    -------
    tuple[str, object]
        ``("xgboost", xgb_model)`` if ``xgb_val_metrics["pr_auc"]`` is
        greater than or equal to ``lgbm_val_metrics["pr_auc"]``, otherwise
        ``("lightgbm", lgbm_model)``.
    """
    if xgb_val_metrics["pr_auc"] >= lgbm_val_metrics["pr_auc"]:
        return "xgboost", xgb_model
    return "lightgbm", lgbm_model


def select_deployed_model(candidates: dict) -> tuple:
    """Pick the overall best-performing trained candidate by validation PR-AUC.

    Unlike `select_champion` (which only compares the two boosted-tree
    candidates), this compares every trained candidate -- including the
    logistic-regression baseline -- so a simpler model that empirically
    wins is not excluded from being the model actually calibrated,
    explained, and deployed.

    Parameters
    ----------
    candidates : dict[str, tuple[object, dict]]
        Mapping of candidate name -> (fitted model, val_metrics dict with
        a "pr_auc" key).

    Returns
    -------
    tuple[str, object]
        The winning candidate's name and fitted model.
    """
    name, (model, _) = max(candidates.items(), key=lambda kv: kv[1][1]["pr_auc"])
    return name, model


def train_isolation_forest(X_train, contamination: float = 0.02) -> IsolationForest:
    """Train an Isolation Forest anomaly detector (unsupervised, no labels).

    Isolation Forest is a complementary unsupervised signal, never trained with
    fraud labels and never used as a supervised classifier. It is trained on
    features alone to detect statistical outliers/anomalies.

    Parameters
    ----------
    X_train : pd.DataFrame or array-like
        Training feature matrix. Must contain no missing values.
    contamination : float, default=0.02
        Expected proportion of anomalies in the dataset (prior for the model).

    Returns
    -------
    IsolationForest
        Fitted Isolation Forest model.
    """
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_train)
    return model


def isolation_forest_anomaly_scores(model: IsolationForest, X) -> np.ndarray:
    """Compute anomaly scores from a fitted Isolation Forest.

    Higher score = more anomalous. Note: sklearn's ``IsolationForest.score_samples``
    returns higher values for *normal* points and lower (more negative) for
    anomalies, so this function negates it to produce the convention:
    higher = more suspicious.

    Parameters
    ----------
    model : IsolationForest
        Fitted Isolation Forest model.
    X : pd.DataFrame or array-like
        Feature matrix to score.

    Returns
    -------
    np.ndarray
        Anomaly scores where higher values indicate more anomalous points.
    """
    return -model.score_samples(X)
