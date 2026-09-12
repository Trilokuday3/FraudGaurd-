from ml.evaluate import evaluate_predictions


def test_evaluate_predictions_perfect_separation():
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.9, 0.95]

    metrics = evaluate_predictions(y_true, y_score, thresholds=(0.5,))

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["thresholds"][0.5]["precision"] == 1.0
    assert metrics["thresholds"][0.5]["recall"] == 1.0


def test_evaluate_predictions_worst_case_inverted_ranking():
    y_true = [0, 1, 0, 1]
    y_score = [0.9, 0.1, 0.8, 0.2]  # exactly inverted: both negatives rank above both positives

    metrics = evaluate_predictions(y_true, y_score, thresholds=(0.5,))

    assert metrics["roc_auc"] == 0.0
    assert metrics["pr_auc"] < 0.5  # worse than the prevalence baseline
