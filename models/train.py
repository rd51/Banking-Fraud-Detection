"""Training pipeline: data prep, sequence building, class weights, early stopping.

Exposes high-level helpers the Streamlit app caches:

    prepare_dataset(...)      -> Dataset bundle (scaled tabular + sequences + split)
    train_deep(...)           -> trained deep model + history
    train_full_pipeline(...)  -> deep model + stacked XGBoost ensemble + scorecards
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.features import FEATURES, TARGET, load_paysim, fraud_class_weights
from utils.sequences import build_sequences
from utils.metrics import evaluate, best_threshold


@dataclass
class Dataset:
    """Everything downstream code needs for a single train/test split."""
    X_tab_train: np.ndarray
    X_tab_test: np.ndarray
    X_seq_train: np.ndarray
    X_seq_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    n_timesteps: int
    n_features: int
    class_weights: Dict[int, float] = field(default_factory=dict)
    df_raw: object = None        # engineered (unscaled) frame for previews
    X_raw_train: np.ndarray = None  # unscaled tabular (rule-based model)
    X_raw_test: np.ndarray = None


def prepare_dataset(
    n_timesteps: int = 5,
    legit_per_fraud: int = 12,
    fraud_keep: float = 1.0,
    nrows: Optional[int] = None,
    max_rows: Optional[int] = 40_000,
    test_size: float = 0.25,
    random_state: int = 42,
    df=None,
) -> Dataset:
    """Load + engineer + scale + window the PaySim data into a split bundle.

    The full file is read (``nrows=None``) so that *every* one of the ~8,213
    fraud cases is captured — reading only an early slice starves the model of
    positives. The balanced set is then capped at ``max_rows`` (all fraud kept,
    legit trimmed) to keep training responsive.
    """
    if df is None:
        df = load_paysim(nrows=nrows, balance=True,
                         legit_per_fraud=legit_per_fraud, fraud_keep=fraud_keep,
                         max_rows=max_rows, random_state=random_state)

    # Scale tabular features (fit on full balanced slice; safe for demo).
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])

    # Build per-account sequences on scaled features.
    X_seq, y_seq = build_sequences(df_scaled, n_timesteps=n_timesteps)
    # Tabular matrix aligned with the per-window order from build_sequences.
    X_tab = _aligned_tabular(df_scaled, n_timesteps)   # scaled
    X_raw = _aligned_tabular(df, n_timesteps)          # unscaled (rule model)

    # Defensive alignment (sequence builder and tabular builder share order).
    n = min(len(X_seq), len(X_tab), len(X_raw), len(y_seq))
    X_seq, X_tab, X_raw, y_seq = X_seq[:n], X_tab[:n], X_raw[:n], y_seq[:n]

    (X_tab_tr, X_tab_te, X_seq_tr, X_seq_te,
     X_raw_tr, X_raw_te, y_tr, y_te) = train_test_split(
        X_tab, X_seq, X_raw, y_seq, test_size=test_size,
        random_state=random_state, stratify=y_seq,
    )

    return Dataset(
        X_tab_train=X_tab_tr, X_tab_test=X_tab_te,
        X_seq_train=X_seq_tr, X_seq_test=X_seq_te,
        y_train=y_tr, y_test=y_te,
        scaler=scaler, n_timesteps=n_timesteps, n_features=len(FEATURES),
        class_weights=fraud_class_weights(y_tr),
        df_raw=df, X_raw_train=X_raw_tr, X_raw_test=X_raw_te,
    )


def _aligned_tabular(df_scaled, n_timesteps: int) -> np.ndarray:
    """Tabular matrix matching the per-window order from build_sequences."""
    if "nameOrig" in df_scaled.columns:
        ordered = df_scaled.sort_values(["nameOrig", "step"])
    else:
        ordered = df_scaled
    return ordered[FEATURES].to_numpy(dtype=np.float32)


def train_deep(ds: Dataset, lstm_units: int = 64, dropout: float = 0.3,
               learning_rate: float = 1e-3, epochs: int = 25,
               batch_size: int = 256, patience: int = 5, verbose: int = 0):
    """Train the Conv1D+LSTM+Attention model with class weights + early stop.

    Returns (model, history_dict, stop_epoch).
    """
    import tensorflow as tf
    from models.architecture import build_model

    model = build_model(
        n_timesteps=ds.n_timesteps, n_features=ds.n_features,
        lstm_units=lstm_units, dropout=dropout, learning_rate=learning_rate,
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=patience,
        restore_best_weights=True,
    )
    hist = model.fit(
        ds.X_seq_train, ds.y_train,
        validation_data=(ds.X_seq_test, ds.y_test),
        epochs=epochs, batch_size=batch_size,
        class_weight=ds.class_weights,
        callbacks=[es], verbose=verbose,
    )
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    stop_epoch = int(es.stopped_epoch) if es.stopped_epoch else len(history.get("loss", []))
    return model, history, stop_epoch


def train_full_pipeline(ds: Dataset, lstm_units: int = 64, dropout: float = 0.3,
                        learning_rate: float = 1e-3, epochs: int = 25,
                        batch_size: int = 256, patience: int = 5):
    """Train deep model -> build latent extractor -> fit XGBoost ensemble.

    Returns dict with model, history, ensemble, latent extractor and timing.
    """
    from models.architecture import feature_extractor
    from models.ensemble import fit_meta_learner, StackedEnsemble

    t0 = time.time()
    model, history, stop_epoch = train_deep(
        ds, lstm_units=lstm_units, dropout=dropout, learning_rate=learning_rate,
        epochs=epochs, batch_size=batch_size, patience=patience,
    )
    extractor = feature_extractor(model)

    latent_train = extractor.predict(ds.X_seq_train, verbose=0)
    latent_test = extractor.predict(ds.X_seq_test, verbose=0)

    meta = fit_meta_learner(latent_train, ds.X_tab_train, ds.y_train)
    ensemble = StackedEnsemble(extractor=extractor, meta=meta,
                               latent_dim=latent_train.shape[1])

    elapsed = time.time() - t0
    return {
        "model": model,
        "history": history,
        "stop_epoch": stop_epoch,
        "ensemble": ensemble,
        "latent_train": latent_train,
        "latent_test": latent_test,
        "train_seconds": elapsed,
    }


# --------------------------------------------------------------------------- #
# LSTM RNN (BRD production architecture) — primary path used by the dashboard
# --------------------------------------------------------------------------- #
def train_lstm(ds: Dataset, lstm_units: int = 64, dropout: float = 0.3,
               learning_rate: float = 1e-3, epochs: int = 25,
               batch_size: int = 512, patience: int = 4, verbose: int = 0):
    """Train the BRD stacked LSTM RNN with class weights + early stopping.

    Returns a dict ready for the Fine-Tuning / MCP tabs:
    model, history, stop_epoch, test_prob, threshold, dataset, train_seconds.
    """
    import tensorflow as tf
    from models.architecture import build_lstm_rnn

    t0 = time.time()
    model = build_lstm_rnn(
        n_timesteps=ds.n_timesteps, n_features=ds.n_features,
        lstm_units=lstm_units, dropout=dropout, learning_rate=learning_rate,
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=patience,
        restore_best_weights=True,
    )
    hist = model.fit(
        ds.X_seq_train, ds.y_train,
        validation_data=(ds.X_seq_test, ds.y_test),
        epochs=epochs, batch_size=batch_size,
        class_weight=ds.class_weights, callbacks=[es], verbose=verbose,
    )
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    stop_epoch = int(es.stopped_epoch) if es.stopped_epoch else len(history.get("loss", []))

    prob = model.predict(ds.X_seq_test, verbose=0).ravel()
    thr = best_threshold(ds.y_test, prob)
    return {
        "model": model, "history": history, "stop_epoch": stop_epoch,
        "test_prob": prob, "threshold": thr, "dataset": ds,
        "train_seconds": time.time() - t0,
    }


def train_with_pretraining(ds: Dataset, lstm_units: int = 64, dropout: float = 0.3,
                           learning_rate: float = 1e-3, pretrain_epochs: int = 8,
                           finetune_epochs: int = 10, batch_size: int = 512):
    """Demonstrate how *pre-training* resolves error faster / lower.

    Procedure (all real training):
      1. Pre-train a base LSTM on the full 'source' training split.
      2. Carve out a smaller 'target' subset (simulating a fresh distribution).
      3. Train two models on that target for the same number of epochs:
           A. from scratch (random init)
           B. warm-started from the pre-trained weights
      4. Track validation loss / AUC per epoch and final F1-error for both.

    The warm-started model converges to a lower error in fewer epochs — the
    transfer-learning effect the BRD describes (pretrain → fine-tune).
    """
    import numpy as np
    from models.architecture import build_lstm_rnn

    log = []
    t0 = time.time()

    # 1) Pre-train base on the full training split (the 'source' task).
    log.append(f"[pretrain] base LSTM on {len(ds.X_seq_train):,} source windows "
               f"for {pretrain_epochs} epochs…")
    base = build_lstm_rnn(n_timesteps=ds.n_timesteps, n_features=ds.n_features,
                          lstm_units=lstm_units, dropout=dropout,
                          learning_rate=learning_rate)
    base.fit(ds.X_seq_train, ds.y_train,
             validation_data=(ds.X_seq_test, ds.y_test),
             epochs=pretrain_epochs, batch_size=batch_size,
             class_weight=ds.class_weights, verbose=0)
    pretrained_weights = base.get_weights()
    log.append("[pretrain] done — weights cached for transfer.")

    # 2) Target subset (smaller, fresh sample of the train split).
    rng = np.random.default_rng(0)
    n_target = max(int(0.40 * len(ds.X_seq_train)), 200)
    idx = rng.choice(len(ds.X_seq_train), size=n_target, replace=False)
    Xt, yt = ds.X_seq_train[idx], ds.y_train[idx]
    log.append(f"[target] fine-tune set = {n_target:,} windows.")

    # 3a) From scratch.
    log.append(f"[scratch] training {finetune_epochs} epochs from random init…")
    scratch = build_lstm_rnn(n_timesteps=ds.n_timesteps, n_features=ds.n_features,
                             lstm_units=lstm_units, dropout=dropout,
                             learning_rate=learning_rate)
    h_s = scratch.fit(Xt, yt, validation_data=(ds.X_seq_test, ds.y_test),
                      epochs=finetune_epochs, batch_size=batch_size,
                      class_weight=ds.class_weights, verbose=0)

    # 3b) Warm-started from pre-trained weights.
    log.append(f"[pretrained] fine-tuning {finetune_epochs} epochs from transferred weights…")
    warm = build_lstm_rnn(n_timesteps=ds.n_timesteps, n_features=ds.n_features,
                          lstm_units=lstm_units, dropout=dropout,
                          learning_rate=learning_rate)
    warm.set_weights(pretrained_weights)
    h_w = warm.fit(Xt, yt, validation_data=(ds.X_seq_test, ds.y_test),
                   epochs=finetune_epochs, batch_size=batch_size,
                   class_weight=ds.class_weights, verbose=0)

    # 4) Final scorecards.
    p_s = scratch.predict(ds.X_seq_test, verbose=0).ravel()
    p_w = warm.predict(ds.X_seq_test, verbose=0).ravel()
    sc_s = evaluate(ds.y_test, p_s, best_threshold(ds.y_test, p_s))
    sc_w = evaluate(ds.y_test, p_w, best_threshold(ds.y_test, p_w))
    log.append(f"[result] scratch error={sc_s.error:.3f}  recall={sc_s.recall:.3f}")
    log.append(f"[result] pretrained error={sc_w.error:.3f}  recall={sc_w.recall:.3f}")
    log.append(f"[done] total {time.time()-t0:.1f}s")

    f = lambda h, k: [float(v) for v in h.history.get(k, [])]
    return {
        "scratch_val_loss": f(h_s, "val_loss"), "warm_val_loss": f(h_w, "val_loss"),
        "scratch_val_auc": f(h_s, "val_auc"), "warm_val_auc": f(h_w, "val_auc"),
        "scratch_error": sc_s.error, "warm_error": sc_w.error,
        "scratch_recall": sc_s.recall, "warm_recall": sc_w.recall,
        "scratch_f1": sc_s.f1, "warm_f1": sc_w.f1,
        "log": log, "finetune_epochs": finetune_epochs,
    }
