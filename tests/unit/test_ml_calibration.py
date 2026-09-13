import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from ml.calibration import calibrate


def test_calibrate_preserves_rank_order_and_bounds():
    rng = np.random.default_rng(0)
    n_train, n_val = 400, 300

    X_train = rng.normal(size=(n_train, 3))
    p_train = 1 / (1 + np.exp(-X_train[:, 0]))
    y_train = (rng.uniform(size=n_train) < p_train).astype(int)
    model = LogisticRegression().fit(X_train, y_train)

    X_val = rng.normal(size=(n_val, 3))
    p_val = 1 / (1 + np.exp(-X_val[:, 0]))
    y_val = (rng.uniform(size=n_val) < p_val).astype(int)

    calibrated = calibrate(model, X_val, y_val, method="isotonic")
    raw_scores = model.predict_proba(X_val)[:, 1]
    calibrated_scores = calibrated.predict_proba(X_val)[:, 1]

    assert calibrated_scores.min() >= 0.0
    assert calibrated_scores.max() <= 1.0
    # Spearman (tie-aware) rank correlation -- isotonic is a monotonic
    # transform so rank order should be highly preserved even though ties
    # can collapse some distinct raw scores to the same calibrated value.
    rho, _ = spearmanr(raw_scores, calibrated_scores)
    assert rho > 0.9
