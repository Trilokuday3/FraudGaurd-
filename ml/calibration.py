"""Probability calibration for the deployed model. Fit on validation data
only -- the deployed model itself was already fit on the training fold, so
this step never touches the test set."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate(
    fitted_model: BaseEstimator, X_val: pd.DataFrame, y_val: pd.Series, method: str = "isotonic"
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    calibrated.fit(X_val, y_val)
    return calibrated
