"""Metric helpers: precision / recall / F1 / ROC-AUC + error-ceiling logic."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)

# Error ceiling from the BRD: all three must clear 0.97 (error <= 3%).
CEILING = 0.97
MAX_ERROR = 0.03


@dataclass
class Scorecard:
    precision: float
    recall: float
    f1: float
    auc: float
    error: float          # 1 - F1
    threshold: float
    n_pos: int
    n_neg: int

    def passes(self) -> bool:
        return (
            self.f1 >= CEILING
            and self.recall >= CEILING
            and self.auc >= CEILING
        )

    def verdict(self) -> str:
        return "DEPLOY" if self.passes() else "REJECT"

    def badge(self) -> str:
        return "🟢 ≤3%" if self.error <= MAX_ERROR else "🔴 >3%"

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate(y_true, y_prob, threshold: float = 0.5) -> Scorecard:
    """Compute a full scorecard at the given probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # AUC needs both classes present.
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    return Scorecard(
        precision=float(p),
        recall=float(r),
        f1=float(f),
        auc=float(auc),
        error=float(1.0 - f),
        threshold=float(threshold),
        n_pos=int(y_true.sum()),
        n_neg=int(len(y_true) - y_true.sum()),
    )


def best_threshold(y_true, y_prob, beta: float = 1.0) -> float:
    """Pick the threshold maximising F-beta on the PR curve."""
    y_true = np.asarray(y_true).astype(int)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]  # align with thresholds
    denom = (beta ** 2 * prec) + rec
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + beta ** 2) * (prec * rec) / np.where(denom == 0, 1, denom)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.nanargmax(fbeta))])


def confusion(y_true, y_prob, threshold: float = 0.5) -> np.ndarray:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])


def roc_points(y_true, y_prob):
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), y_prob)
    return fpr, tpr


def pr_points(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(np.asarray(y_true).astype(int), y_prob)
    ap = average_precision_score(np.asarray(y_true).astype(int), y_prob)
    return rec, prec, ap


def weighted_cross_entropy(y_true, y_prob, w1: float, w0: float, eps: float = 1e-7):
    """L = -mean[w1*y*log(p) + w0*(1-y)*log(1-p)]  (vectorised)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(w1 * y * np.log(p) + w0 * (1 - y) * np.log(1 - p)))
