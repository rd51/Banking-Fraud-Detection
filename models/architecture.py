"""Primary deep-learning architecture (Keras functional API).

    Input (N_TIMESTEPS, N_FEATURES)
      -> Conv1D(64, k=3, s=1, relu)        fine-grained patterns
      -> Conv1D(32, k=3, s=2, relu)        downsampled coarse patterns
      -> LSTM(64, return_sequences=True)
      -> LSTM(32, return_sequences=True)
      -> MultiHeadAttention(heads=4, key_dim=16) + residual + LayerNorm
      -> Position-wise FFN + residual + LayerNorm
      -> GlobalAveragePooling1D
      -> Dense(32, relu)                    <-- 32-dim latent feature vector
      -> Dense(1, sigmoid)                  classification head

The Dense(32) layer is named ``latent`` so the ensemble can tap it as a
feature extractor for the XGBoost meta-learner. The attention layer returns
its scores so the dashboard can render the attention heatmap.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.optimizers import Adam
    _TF_AVAILABLE = True
except Exception:  # pragma: no cover - allows import without TF for docs
    _TF_AVAILABLE = False

N_TIMESTEPS = 5
N_FEATURES = 8
LATENT_DIM = 32


def conv_out_len(l_in: int, kernel: int, stride: int) -> int:
    """L_out = floor((L_in - kernel) / stride) + 1  (valid padding)."""
    return (l_in - kernel) // stride + 1


def build_lstm_rnn(
    n_timesteps: int = N_TIMESTEPS,
    n_features: int = N_FEATURES,
    lstm_units: int = 64,
    lstm_units_2: int | None = None,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    """BRD/TRD production architecture — pure stacked LSTM RNN.

        Input -> Masking(0.0)
              -> LSTM(64, tanh, return_sequences=True) -> Dropout(0.3)
              -> LSTM(32, tanh)                        -> Dropout(0.3)
              -> Dense(16, relu)
              -> Dense(1, sigmoid)

    Two stacked LSTM layers: the first returns a sequence to the second, which
    compresses it to a single vector — capturing both local transaction-to-
    transaction patterns and global account-level trajectories.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the model.")

    lstm_units_2 = lstm_units_2 or max(lstm_units // 2, 16)

    inp = Input(shape=(n_timesteps, n_features), name="sequence_input")
    x = layers.Masking(mask_value=0.0, name="masking")(inp)
    x = layers.LSTM(lstm_units, activation="tanh", return_sequences=True,
                    name="lstm_1")(x)
    x = layers.Dropout(dropout, name="drop_1")(x)
    x = layers.LSTM(lstm_units_2, activation="tanh", name="lstm_2")(x)
    x = layers.Dropout(dropout, name="drop_2")(x)
    x = layers.Dense(16, activation="relu", name="dense_16")(x)
    out = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)

    model = Model(inp, out, name="lstm_rnn")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )
    return model


def lstm_rnn_summary(lstm_units: int = 64, dropout: float = 0.3) -> str:
    """Plain-text layer table for the architecture tab (TF-independent)."""
    u2 = max(lstm_units // 2, 16)
    return (
        f"Input            : ({N_TIMESTEPS}, {N_FEATURES})\n"
        f"Masking(0.0)     : ({N_TIMESTEPS}, {N_FEATURES})\n"
        f"LSTM({lstm_units}, seq)    : ({N_TIMESTEPS}, {lstm_units})\n"
        f"Dropout({dropout})     : ({N_TIMESTEPS}, {lstm_units})\n"
        f"LSTM({u2})         : ({u2},)\n"
        f"Dropout({dropout})     : ({u2},)\n"
        f"Dense(16, relu)  : (16,)\n"
        f"Dense(1, sigmoid): (1,)   -> fraud probability"
    )


def transformer_block(x, num_heads: int = 4, key_dim: int = 16, ff_dim: int = 64,
                      dropout: float = 0.1, name: str = "attn"):
    """Multi-head self-attention + residual/LN + position-wise FFN + residual/LN.

    Returns (output_tensor, attention_layer) so callers can later request the
    attention scores via a side-model.
    """
    mha = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_mha"
    )
    attn_out, attn_scores = mha(x, x, return_attention_scores=True)
    attn_out = layers.Dropout(dropout)(attn_out)
    x1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x + attn_out)

    # Position-wise feed-forward network: FFN(x) = max(0, xW1+b1)W2+b2
    ffn = layers.Dense(ff_dim, activation="relu", name=f"{name}_ffn1")(x1)
    ffn = layers.Dense(x1.shape[-1], name=f"{name}_ffn2")(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    x2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x1 + ffn)
    return x2, attn_scores


def build_model(
    n_timesteps: int = N_TIMESTEPS,
    n_features: int = N_FEATURES,
    lstm_units: int = 64,
    lstm_units_2: int | None = None,
    dropout: float = 0.3,
    num_heads: int = 4,
    key_dim: int = 16,
    learning_rate: float = 1e-3,
    return_attention: bool = False,
):
    """Construct and compile the full architecture.

    When ``return_attention`` is True a second model exposing the attention
    score tensor is also returned (useful for the dashboard heatmaps).
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the model.")

    lstm_units_2 = lstm_units_2 or max(lstm_units // 2, 16)

    inp = Input(shape=(n_timesteps, n_features), name="sequence_input")

    # --- Conv1D feature extraction -------------------------------------- #
    x = layers.Conv1D(64, 3, strides=1, activation="relu", padding="valid",
                      name="conv1d_fine")(inp)
    # Guard against the sequence collapsing to length < kernel for stride 2.
    if conv_out_len(conv_out_len(n_timesteps, 3, 1), 3, 2) >= 1:
        x = layers.Conv1D(32, 3, strides=2, activation="relu", padding="valid",
                          name="conv1d_coarse")(x)
    else:  # very short windows -> keep stride 1 so length stays >= 1
        x = layers.Conv1D(32, 3, strides=1, activation="relu", padding="same",
                          name="conv1d_coarse")(x)

    # --- Recurrent stack ------------------------------------------------- #
    x = layers.LSTM(lstm_units, return_sequences=True, activation="tanh",
                    name="lstm_1")(x)
    x = layers.Dropout(dropout, name="drop_lstm1")(x)
    x = layers.LSTM(lstm_units_2, return_sequences=True, activation="tanh",
                    name="lstm_2")(x)
    x = layers.Dropout(dropout, name="drop_lstm2")(x)

    # --- Transformer encoder block -------------------------------------- #
    x, attn_scores = transformer_block(
        x, num_heads=num_heads, key_dim=key_dim, ff_dim=lstm_units_2 * 2,
        dropout=dropout, name="encoder",
    )

    # --- Latent vector + head ------------------------------------------- #
    pooled = layers.GlobalAveragePooling1D(name="gap")(x)
    latent = layers.Dense(LATENT_DIM, activation="relu", name="latent")(pooled)
    latent = layers.Dropout(dropout, name="drop_latent")(latent)
    out = layers.Dense(1, activation="sigmoid", name="fraud_prob")(latent)

    model = Model(inp, out, name="conv_lstm_attention")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )

    if return_attention:
        attn_model = Model(inp, attn_scores, name="attention_probe")
        return model, attn_model
    return model


def feature_extractor(model) -> "Model":
    """Return a model mapping input -> 32-dim latent vector (pre-dropout)."""
    return Model(model.input, model.get_layer("latent").output,
                 name="latent_extractor")


def attention_probe(model) -> "Model":
    """Return a model mapping input -> attention scores (heads, q, k)."""
    # Re-resolve the attention score tensor from the named MHA layer.
    mha_layer = model.get_layer("encoder_mha")
    # Rebuild a probe using the layer's stored input.
    lstm2_out = model.get_layer("drop_lstm2").output
    _, scores = mha_layer(lstm2_out, lstm2_out, return_attention_scores=True)
    return Model(model.input, scores, name="attention_probe")


def architecture_summary() -> str:
    """Plain-text summary string used by the dashboard if TF is unavailable."""
    l1 = conv_out_len(N_TIMESTEPS, 3, 1)
    l2 = conv_out_len(l1, 3, 2)
    return (
        f"Input            : ({N_TIMESTEPS}, {N_FEATURES})\n"
        f"Conv1D(64,k3,s1) : ({l1}, 64)\n"
        f"Conv1D(32,k3,s2) : ({max(l2,1)}, 32)\n"
        f"LSTM(64, seq)    : ({max(l2,1)}, 64)\n"
        f"LSTM(32, seq)    : ({max(l2,1)}, 32)\n"
        f"MHA(4 x 16)+LN   : ({max(l2,1)}, 32)\n"
        f"FFN + LN         : ({max(l2,1)}, 32)\n"
        f"GlobalAvgPool    : (32,)\n"
        f"Dense latent(32) : (32,)\n"
        f"Dense sigmoid    : (1,)"
    )
