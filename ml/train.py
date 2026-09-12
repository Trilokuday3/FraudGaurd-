"""Model training: baseline -> Random Forest -> XGBoost/LightGBM champion ->
Isolation Forest. Every classifier here uses class-weighting for the ~1.5%
imbalance, never resampling -- see the spec's "Why time-based split" and
"Class imbalance handling" sections for the reasoning."""

from sklearn.ensemble import RandomForestClassifier
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
