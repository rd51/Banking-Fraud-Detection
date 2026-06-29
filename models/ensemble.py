"""XGBoost stacking meta-learner on LSTM latent features (Model 4, PRIMARY).

Stacking pipeline
-----------------
    1. Deep base (Conv1D+LSTM+Attention) trained end-to-end.
    2. Latent extractor taps the Dense(32) ``latent`` layer.
    3. Meta-feature matrix = concat([latent (32-dim), raw tabular (8-dim)]).
    4. XGBoost meta-learner produces the final fraud probability.

    L = Σ l(ŷ_i, y_i) + Σ_k Ω(f_k),   Ω(f) = γT + ½λ‖w‖²
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from xgboost import XGBClassifier

from utils.features import FEATURES, scale_pos_weight


# Names for the combined meta-feature space (used in importance charts).
def meta_feature_names(latent_dim: int = 32) -> List[str]:
    return [f"latent_{i}" for i in range(latent_dim)] + list(FEATURES)


@dataclass
class StackedEnsemble:
    """Holds the deep latent extractor + the XGBoost meta-learner."""

    extractor: object          # keras Model: input -> latent (32-dim)
    meta: XGBClassifier
    latent_dim: int = 32

    def _meta_features(self, X_seq: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        latent = self.extractor.predict(X_seq, verbose=0)
        return np.hstack([latent, X_tab])

    def predict_proba(self, X_seq: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        feats = self._meta_features(X_seq, X_tab)
        return self.meta.predict_proba(feats)[:, 1]

    def feature_importance(self) -> np.ndarray:
        return self.meta.feature_importances_


def fit_meta_learner(latent_train: np.ndarray, X_tab_train: np.ndarray,
                     y_train: np.ndarray, n_estimators: int = 400,
                     max_depth: int = 5, learning_rate: float = 0.05,
                     reg_lambda: float = 1.0, gamma: float = 0.0,
                     random_state: int = 42) -> XGBClassifier:
    """Train the XGBoost meta-learner on stacked latent + tabular features."""
    meta_X = np.hstack([latent_train, X_tab_train])
    spw = scale_pos_weight(y_train)
    meta = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        gamma=gamma,
        scale_pos_weight=spw,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )
    meta.fit(meta_X, y_train)
    return meta


def build_ensemble(deep_model, latent_train, X_tab_train, y_train,
                   **meta_kwargs) -> StackedEnsemble:
    """Convenience: build extractor from a trained deep model + fit meta."""
    from models.architecture import feature_extractor

    extractor = feature_extractor(deep_model)
    meta = fit_meta_learner(latent_train, X_tab_train, y_train, **meta_kwargs)
    return StackedEnsemble(extractor=extractor, meta=meta,
                           latent_dim=latent_train.shape[1])
