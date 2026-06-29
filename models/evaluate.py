"""Evaluation helpers: threshold tuning, confusion matrices, speed benchmarks."""

from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np

from utils.metrics import Scorecard, evaluate, best_threshold


def evaluate_probs(y_true, y_prob, tune: bool = True,
                   beta: float = 1.0) -> Scorecard:
    """Score predictions, optionally tuning the threshold for F-beta."""
    thr = best_threshold(y_true, y_prob, beta=beta) if tune else 0.5
    return evaluate(y_true, y_prob, threshold=thr)


def benchmark_inference(predict_fn: Callable[[np.ndarray], np.ndarray],
                        X: np.ndarray, n: int = 1000, repeats: int = 3) -> float:
    """Return inference time per 1000 transactions in milliseconds."""
    n = min(n, len(X))
    if n == 0:
        return 0.0
    sample = X[:n]
    # warm-up
    predict_fn(sample)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(sample)
        times.append(time.perf_counter() - t0)
    per_n = float(np.median(times))
    return per_n / n * 1000.0 * 1000.0  # seconds/n -> ms per 1000


def rule_based_scores(raw_tab) -> np.ndarray:
    """Model 1: pure threshold logic, returns pseudo-probabilities in {0,1}.

    flag if (amount > 10000 AND type == TRANSFER) OR errorBalanceOrig > amount

    ``raw_tab`` is the *unscaled* tabular matrix in FEATURES order:
    [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
     errorBalanceOrig, errorBalanceDest, type_TRANSFER]
    or a DataFrame with those columns.
    """
    if hasattr(raw_tab, "to_numpy"):
        amount = raw_tab["amount"].to_numpy()
        is_transfer = raw_tab["type_TRANSFER"].to_numpy() == 1
        err_orig = raw_tab["errorBalanceOrig"].to_numpy()
    else:
        arr = np.asarray(raw_tab)
        amount = arr[:, 0]
        err_orig = arr[:, 5]
        is_transfer = arr[:, 7] == 1
    flag = ((amount > 10000) & is_transfer) | (err_orig > amount)
    return flag.astype(float)


def decision_route(prob: float) -> str:
    """CLEAR (<0.30) / REVIEW (0.30-0.70) / FREEZE (>0.70)."""
    if prob < 0.30:
        return "CLEAR"
    if prob <= 0.70:
        return "REVIEW"
    return "FREEZE"
