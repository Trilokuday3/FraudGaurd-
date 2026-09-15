import numpy as np

from decision.select_thresholds import compute_total_cost, grid_search_thresholds


def test_compute_total_cost_all_approve_charges_only_false_negatives():
    scores = np.array([0.1, 0.2, 0.1, 0.2])
    labels = np.array([0, 1, 0, 1])
    amounts = np.array([50.0, 100.0, 20.0, 200.0])

    cost = compute_total_cost(scores, labels, amounts, t_review=0.9, t_block=0.95)

    # both fraud rows (amounts 100, 200) are approved -> lose their full amount
    assert cost == 300.0


def test_compute_total_cost_all_block_charges_only_false_positives():
    scores = np.array([0.1, 0.2, 0.1, 0.2])
    labels = np.array([0, 1, 0, 1])
    amounts = np.array([50.0, 100.0, 20.0, 200.0])

    cost = compute_total_cost(scores, labels, amounts, t_review=0.0, t_block=0.0)

    # both legit rows (labels 0) get blocked -> false positive cost each
    assert cost == 50.0  # 2 * 25.0


def test_compute_total_cost_review_charges_flat_fee_regardless_of_label():
    scores = np.array([0.5, 0.5])
    labels = np.array([0, 1])
    amounts = np.array([1000.0, 1000.0])

    cost = compute_total_cost(scores, labels, amounts, t_review=0.3, t_block=0.9)

    # both rows land in the review band -> flat review cost each, not the
    # amount, even for the fraud row
    assert cost == 7.0  # 2 * 3.50


def test_grid_search_finds_the_zero_cost_perfect_split():
    # fraud rows score high (0.95), legit rows score low (0.05) -- a perfect
    # split exists (approve everything below 0.10, block everything at or
    # above 0.95), which achieves zero total cost.
    scores = np.array([0.05, 0.95, 0.05, 0.95, 0.05])
    labels = np.array([0, 1, 0, 1, 0])
    amounts = np.array([40.0, 500.0, 30.0, 600.0, 20.0])

    best = grid_search_thresholds(scores, labels, amounts, step=0.05)

    assert best["total_cost"] == 0.0
    assert best["t_review"] > 0.05
    assert best["t_block"] <= 0.95

    assert "cost_curve" in best
    assert len(best["cost_curve"]) > 0
    assert all("t_review" in point and "cost" in point for point in best["cost_curve"])
