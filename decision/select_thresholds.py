"""Cost-sensitive threshold selection: grid-search (t_review, t_block) over
the validation split to minimize total realized cost under the cost matrix
in docs/superpowers/specs/2026-09-15-decision-engine-api-design.md."""

import json
import os

import mlflow
import mlflow.sklearn
import numpy as np

from ml.data import load_features_and_labels, prepare_model_matrix, time_based_split

REVIEW_COST = 3.50
FALSE_POSITIVE_COST = 25.0


def compute_total_cost(
    scores,
    labels,
    amounts,
    t_review: float,
    t_block: float,
    review_cost: float = REVIEW_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> float:
    """Total realized cost of a (t_review, t_block) policy on labeled data.

    A false negative (approved fraud) costs the transaction's own amount;
    any review costs a flat fee regardless of the true label; a false
    positive (blocked legitimate transaction) costs a flat friction fee;
    correctly approved or correctly blocked/reviewed-and-caught fraud costs
    nothing beyond the review fee already counted.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    amounts = np.asarray(amounts)

    is_review = (scores >= t_review) & (scores < t_block)
    is_block = scores >= t_block
    is_approve = scores < t_review

    false_negative_cost = np.where(is_approve & (labels == 1), amounts, 0.0).sum()
    review_total_cost = np.where(is_review, review_cost, 0.0).sum()
    false_positive_total_cost = np.where(is_block & (labels == 0), false_positive_cost, 0.0).sum()

    return float(false_negative_cost + review_total_cost + false_positive_total_cost)


def grid_search_thresholds(scores, labels, amounts, step: float = 0.01) -> dict:
    """Grid-search t_review < t_block over [0, 1] and return the minimum-cost pair."""
    candidates = np.arange(0.0, 1.0 + step, step)
    best = {"t_review": 0.0, "t_block": 1.0, "total_cost": float("inf")}

    for t_review in candidates:
        for t_block in candidates:
            if t_block <= t_review:
                continue
            cost = compute_total_cost(scores, labels, amounts, t_review, t_block)
            if cost < best["total_cost"]:
                best = {
                    "t_review": round(float(t_review), 4),
                    "t_block": round(float(t_block), 4),
                    "total_cost": cost,
                }

    cost_curve = [
        {
            "t_review": round(float(t_review), 4),
            "cost": compute_total_cost(scores, labels, amounts, t_review, best["t_block"]),
        }
        for t_review in candidates
        if t_review < best["t_block"]
    ]
    best["cost_curve"] = cost_curve

    return best


def select_thresholds_for_run(
    run_id: str,
    data_dir: str = "./data",
    mlflow_tracking_uri: str = "./mlruns",
    step: float = 0.01,
    output_path: str = "./decision/thresholds.json",
) -> dict:
    """Load model from MLflow run, score validation split, grid-search thresholds,
    write to output_path, and return the result dict.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the 'calibrated_deployed_model'.
    data_dir : str, optional
        Path to data directory, by default "./data".
    mlflow_tracking_uri : str, optional
        MLflow tracking URI, by default "./mlruns".
    step : float, optional
        Grid search step size, by default 0.01.
    output_path : str, optional
        Path to write thresholds.json, by default "./decision/thresholds.json".

    Returns
    -------
    dict
        Result dictionary with keys: t_review, t_block, total_cost, model_run_id,
        cost_matrix.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")

    features, labels = load_features_and_labels(data_dir)
    (_, _), (val_X_raw, val_y), _ = time_based_split(features, labels)
    val_X = prepare_model_matrix(val_X_raw)
    val_scores = model.predict_proba(val_X)[:, 1]

    best = grid_search_thresholds(
        val_scores, val_y.to_numpy(), val_X_raw["amount"].to_numpy(), step=step
    )

    result = {
        **best,
        "model_run_id": run_id,
        "cost_matrix": {
            "review_cost": REVIEW_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": "transaction amount",
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
