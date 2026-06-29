"""Sliding-window sequence builder + zero padding.

Each account (nameOrig) generates overlapping windows of its last
``n_timesteps`` transactions. This is the temporal input the Conv1D/LSTM
stack consumes.

Complexity
----------
Sliding window build:  Time O(n)   Space O(n * k)   (n rows, k = window size)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .features import FEATURES, TARGET


def build_sequences(
    df: pd.DataFrame,
    n_timesteps: int = 5,
    features: List[str] | None = None,
    group_col: str = "nameOrig",
    sort_col: str = "step",
    label_last: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct per-account sliding windows.

    For each account ordered by ``step``, slide a window of length
    ``n_timesteps`` across its transactions. Windows shorter than the required
    length (early transactions) are left-padded with zeros. The label of a
    window is the fraud flag of its **last** transaction (the one being scored).

    Returns
    -------
    X : (n_windows, n_timesteps, n_features) float32
    y : (n_windows,) int8
    """
    features = features or FEATURES
    n_feat = len(features)

    # Fall back to a single global ordering if grouping column is absent.
    if group_col not in df.columns:
        return _global_windows(df, n_timesteps, features, label_last)

    df = df.sort_values([group_col, sort_col])
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    feat_vals = df[features].to_numpy(dtype=np.float32)
    labels = df[TARGET].to_numpy(dtype=np.int8) if TARGET in df else np.zeros(len(df), np.int8)
    groups = df[group_col].to_numpy()

    # Identify contiguous group boundaries (data already sorted by group).
    boundaries = np.where(groups[1:] != groups[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(groups)]])

    for s, e in zip(starts, ends):
        block = feat_vals[s:e]
        block_lab = labels[s:e]
        for t in range(len(block)):
            lo = max(0, t - n_timesteps + 1)
            window = block[lo : t + 1]
            if len(window) < n_timesteps:  # left-pad with zeros
                pad = np.zeros((n_timesteps - len(window), n_feat), np.float32)
                window = np.vstack([pad, window])
            X_list.append(window)
            y_list.append(int(block_lab[t]) if label_last else int(block_lab[max(0, t)]))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int8)
    return X, y


def _global_windows(df, n_timesteps, features, label_last):
    """Window builder when no account grouping is available."""
    feat_vals = df[features].to_numpy(dtype=np.float32)
    labels = df[TARGET].to_numpy(dtype=np.int8) if TARGET in df else np.zeros(len(df), np.int8)
    n_feat = len(features)
    X_list, y_list = [], []
    for t in range(len(feat_vals)):
        lo = max(0, t - n_timesteps + 1)
        window = feat_vals[lo : t + 1]
        if len(window) < n_timesteps:
            pad = np.zeros((n_timesteps - len(window), n_feat), np.float32)
            window = np.vstack([pad, window])
        X_list.append(window)
        y_list.append(int(labels[t]))
    return np.asarray(X_list, np.float32), np.asarray(y_list, np.int8)


def pad_single_sequence(
    rows: np.ndarray, n_timesteps: int = 5
) -> np.ndarray:
    """Pad/truncate one (t, n_features) array to (1, n_timesteps, n_features)."""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    n_feat = rows.shape[1]
    if len(rows) >= n_timesteps:
        window = rows[-n_timesteps:]
    else:
        pad = np.zeros((n_timesteps - len(rows), n_feat), np.float32)
        window = np.vstack([pad, rows])
    return window.reshape(1, n_timesteps, n_feat)


def sliding_window_pseudocode() -> str:
    """Pseudocode shown in the DSA tab."""
    return (
        "function build_windows(transactions T, k):\n"
        "    windows = []                      # O(1)\n"
        "    for t in 0 .. len(T)-1:           # O(n)\n"
        "        lo = max(0, t - k + 1)\n"
        "        window = T[lo .. t]\n"
        "        if len(window) < k:\n"
        "            window = zero_pad_left(window, k)   # O(k)\n"
        "        windows.append(window)        # amortised O(1)\n"
        "    return windows                    # Total: Time O(n), Space O(n*k)"
    )
