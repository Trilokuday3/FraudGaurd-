import numpy as np

from ml.train import isolation_forest_anomaly_scores, train_isolation_forest


def test_isolation_forest_flags_obvious_outlier():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0, scale=1, size=(200, 3))
    outlier = np.array([[50, 50, 50]])
    X = np.vstack([normal, outlier])

    model = train_isolation_forest(X, contamination=0.01)
    scores = isolation_forest_anomaly_scores(model, X)

    assert scores[-1] > scores[:-1].max()
