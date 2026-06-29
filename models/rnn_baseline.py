"""Simple RNN baseline (Model 3 in the battle).

    Masking -> SimpleRNN(32, tanh) -> Dropout(0.3) -> Dense(16) -> Dense(1, sigmoid)

Deliberately weak: SimpleRNN suffers vanishing gradients on longer windows and
is included only as a baseline reference, never a deployment candidate.
"""

from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.optimizers import Adam
    _TF_AVAILABLE = True
except Exception:  # pragma: no cover
    _TF_AVAILABLE = False


def build_rnn_baseline(n_timesteps: int = 5, n_features: int = 8,
                       learning_rate: float = 1e-3):
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for the RNN baseline.")

    inp = Input(shape=(n_timesteps, n_features), name="seq_in")
    x = layers.Masking(mask_value=0.0, name="mask")(inp)
    x = layers.SimpleRNN(32, activation="tanh", name="simple_rnn")(x)
    x = layers.Dropout(0.3, name="drop")(x)
    x = layers.Dense(16, activation="relu", name="dense")(x)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = Model(inp, out, name="simple_rnn_baseline")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
