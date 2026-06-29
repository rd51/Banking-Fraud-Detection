"""Feature-engineering pipeline for the PaySim dataset.

Implements exactly the transformations defined in the BRD/TRD:

    errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest
    type_TRANSFER    = (type == 'TRANSFER').astype(int)

Only TRANSFER and CASH_OUT transactions carry fraud in PaySim, so the loader
filters to those two types.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

# Canonical feature ordering used everywhere (model input, XGBoost, scaler).
FEATURES: List[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "type_TRANSFER",
]

TARGET = "isFraud"

# Candidate locations for the dataset (root, ./data, env override).
_CSV_CANDIDATES = [
    os.environ.get("PAYSIM_CSV", ""),
    os.path.join("data", "paysim.csv"),
    "paysim.csv",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "paysim.csv"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "paysim.csv"),
]


def find_csv() -> Optional[str]:
    """Return the first existing PaySim CSV path, or None if none found."""
    for path in _CSV_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return None


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply BRD feature engineering and return a clean, filtered frame.

    Keeps only TRANSFER / CASH_OUT rows, adds balance-error features and the
    TRANSFER one-hot flag, and guarantees all model features exist.
    """
    df = df.copy()

    # Normalise column names that vary across PaySim exports.
    rename = {
        "oldbalanceOrg": "oldbalanceOrg",
        "newbalanceOrig": "newbalanceOrig",
        "oldbalanceDest": "oldbalanceDest",
        "newbalanceDest": "newbalanceDest",
    }
    df = df.rename(columns=rename)

    # Filter to fraud-bearing transaction types.
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    # Balance-error features (capture impossible / inconsistent balances).
    df["errorBalanceOrig"] = (
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)

    # Defensive: replace inf / NaN introduced by malformed rows.
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def load_paysim(
    path: Optional[str] = None,
    nrows: Optional[int] = None,
    balance: bool = False,
    fraud_keep: float = 1.0,
    legit_per_fraud: int = 12,
    max_rows: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and engineer the PaySim dataset.

    Parameters
    ----------
    path : explicit CSV path; auto-detected when None.
    nrows : cap on rows read from disk (the file is ~6.3M rows / 493 MB).
        Pass None to read the whole file — needed for training so that every
        one of the ~8,213 fraud cases is captured, not just those in an early
        slice.
    balance : if True, down-sample legit rows to ``legit_per_fraud`` per fraud
        so in-app training stays fast while keeping every fraud case.
    max_rows : optional cap on the balanced set size (keeps all fraud, trims
        legit) to bound training time.
    fraud_keep : fraction of fraud rows to retain (1.0 = all).
    """
    if path is None:
        path = find_csv()
    if path is None:
        raise FileNotFoundError(
            "paysim.csv not found. Place it in ./data/ or upload via the app."
        )

    usecols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
    ]

    # Full-file balanced load: stream in chunks so peak memory stays small while
    # still capturing every fraud case across the whole 493 MB file.
    if balance and nrows is None:
        df = _load_balanced_chunked(path, usecols, legit_per_fraud, fraud_keep,
                                    max_rows, random_state)
        return df.reset_index(drop=True)

    df = pd.read_csv(path, usecols=lambda c: c in usecols, nrows=nrows)
    df = engineer(df)
    if balance:
        df = downsample(df, legit_per_fraud, fraud_keep, max_rows, random_state)
    return df.reset_index(drop=True)


def _load_balanced_chunked(
    path: str,
    usecols: list,
    legit_per_fraud: int,
    fraud_keep: float,
    max_rows: Optional[int],
    random_state: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream the CSV, keeping ALL fraud rows + a bounded legit sample.

    Memory stays ~O(chunksize + kept rows) instead of holding all 6.3M rows.
    """
    target_total = max_rows or 30_000
    # Legit to sample per chunk so the pool comfortably covers the final need.
    n_chunks_est = 14  # ~6.36M rows / 500k
    legit_per_chunk = max(target_total // n_chunks_est * 2, 2_000)

    fraud_parts, legit_parts = [], []
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols,
                             chunksize=chunksize):
        chunk = chunk[chunk["type"].isin(["TRANSFER", "CASH_OUT"])]
        if chunk.empty:
            continue
        fraud_parts.append(chunk[chunk["isFraud"] == 1])
        legit_chunk = chunk[chunk["isFraud"] == 0]
        if len(legit_chunk) > legit_per_chunk:
            legit_chunk = legit_chunk.sample(n=legit_per_chunk,
                                             random_state=random_state)
        legit_parts.append(legit_chunk)

    fraud = pd.concat(fraud_parts) if fraud_parts else pd.DataFrame()
    legit_pool = pd.concat(legit_parts) if legit_parts else pd.DataFrame()

    if fraud_keep < 1.0 and len(fraud):
        fraud = fraud.sample(frac=fraud_keep, random_state=random_state)

    n_legit = min(len(legit_pool), int(len(fraud) * legit_per_fraud))
    if max_rows is not None:
        n_legit = min(n_legit, max(max_rows - len(fraud), len(fraud)))
    legit = legit_pool.sample(n=n_legit, random_state=random_state)

    out = pd.concat([fraud, legit]).sample(frac=1.0, random_state=random_state)
    return engineer(out)


def downsample(
    df: pd.DataFrame,
    legit_per_fraud: int = 12,
    fraud_keep: float = 1.0,
    max_rows: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep all (or a fraction of) fraud rows + N legit rows per fraud.

    If ``max_rows`` is given and the balanced set would exceed it, the legit
    pool is trimmed further (all fraud is always retained).
    """
    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]

    if fraud_keep < 1.0:
        fraud = fraud.sample(frac=fraud_keep, random_state=random_state)

    n_legit = min(len(legit), int(len(fraud) * legit_per_fraud))
    if max_rows is not None:
        n_legit = min(n_legit, max(max_rows - len(fraud), len(fraud)))
    legit = legit.sample(n=n_legit, random_state=random_state)

    out = pd.concat([fraud, legit]).sample(frac=1.0, random_state=random_state)
    return out.reset_index(drop=True)


def fraud_class_weights(y: np.ndarray) -> dict:
    """Balanced class weights w1 = N/(2*N_fraud), w0 = N/(2*N_legit)."""
    y = np.asarray(y)
    n = len(y)
    n_fraud = max(int(y.sum()), 1)
    n_legit = max(n - n_fraud, 1)
    return {0: n / (2.0 * n_legit), 1: n / (2.0 * n_fraud)}


def scale_pos_weight(y: np.ndarray) -> float:
    """XGBoost scale_pos_weight = N_legit / N_fraud."""
    y = np.asarray(y)
    n_fraud = max(int(y.sum()), 1)
    n_legit = max(len(y) - n_fraud, 1)
    return float(n_legit) / float(n_fraud)
