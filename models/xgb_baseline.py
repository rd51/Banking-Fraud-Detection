"""XGBoost standalone baseline (Model 2 in the battle).

Pure tabular model on the 8 engineered features — no temporal modelling.
Serves both as a battle competitor and as a sanity check on feature quality.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from utils.features import scale_pos_weight


def build_xgb_baseline(y_train: np.ndarray, n_estimators: int = 300,
                       max_depth: int = 6, learning_rate: float = 0.05,
                       random_state: int = 42) -> XGBClassifier:
    """Instantiate the standalone XGBoost classifier with imbalance weighting."""
    spw = scale_pos_weight(y_train)
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )


def train_xgb_baseline(X_train, y_train, **kwargs) -> XGBClassifier:
    model = build_xgb_baseline(y_train, **kwargs)
    model.fit(X_train, y_train)
    return model
