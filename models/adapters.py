"""Transfer-learning adapter layers (parameter-efficient fine-tuning).

A frozen Conv1D+LSTM base (conceptually pretrained on IEEE-CIS) is reused and
small trainable bottleneck adapters are inserted:

    h' = h + W_up · ReLU(W_down · h)
         W_down : d -> d/4      (bottleneck down-projection)
         W_up   : d/4 -> d      (up-projection back to model dim)

Two adapter *variants* target different tasks:
    * fraud-detection adapter
    * AML-layering adapter

Only the adapters + classification head are trainable; the base is frozen, so
fine-tuning on PaySim touches a tiny fraction of the parameters.
"""

from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.optimizers import Adam
    _TF_AVAILABLE = True
except Exception:  # pragma: no cover
    _TF_AVAILABLE = False


class Adapter(layers.Layer):
    """Bottleneck residual adapter:  h' = h + W_up(ReLU(W_down(h)))."""

    def __init__(self, d_model: int, reduction: int = 4, name: str = "adapter",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.bottleneck = max(d_model // reduction, 1)
        self.down = layers.Dense(self.bottleneck, activation="relu",
                                 name=f"{name}_down")
        self.up = layers.Dense(d_model, name=f"{name}_up")

    def call(self, x):
        return x + self.up(self.down(x))  # residual connection

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


def freeze_base(model, until_layer: str = "drop_lstm2") -> None:
    """Freeze every layer up to and including ``until_layer`` (the base)."""
    freezing = True
    for layer in model.layers:
        layer.trainable = freezing
        if layer.name == until_layer:
            freezing = False
    # Ensure frozen flag actually applied to base.
    hit = False
    for layer in model.layers:
        if not hit:
            layer.trainable = False
        if layer.name == until_layer:
            hit = True


def build_adapted_model(base_model, task: str = "fraud", reduction: int = 4,
                        learning_rate: float = 5e-4):
    """Insert a task-specific adapter after the frozen LSTM base and recompile.

    Parameters
    ----------
    base_model : a model from architecture.build_model (its base is frozen).
    task : 'fraud' or 'aml' — selects the adapter variant / head name.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for adapter fine-tuning.")

    # Freeze the convolutional + recurrent base.
    freeze_base(base_model, until_layer="drop_lstm2")

    seq_out = base_model.get_layer("drop_lstm2").output   # (T, d_model)
    d_model = seq_out.shape[-1]

    adapter = Adapter(d_model, reduction=reduction, name=f"{task}_adapter")
    h = adapter(seq_out)
    h = layers.GlobalAveragePooling1D(name=f"{task}_gap")(h)
    h = layers.Dense(32, activation="relu", name=f"{task}_latent")(h)
    out = layers.Dense(1, activation="sigmoid", name=f"{task}_head")(h)

    adapted = Model(base_model.input, out, name=f"adapted_{task}")
    adapted.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return adapted


def adapter_param_summary(d_model: int = 32, reduction: int = 4) -> dict:
    """Return parameter counts to demonstrate efficiency vs full fine-tune."""
    bottleneck = max(d_model // reduction, 1)
    down = d_model * bottleneck + bottleneck   # weights + bias
    up = bottleneck * d_model + d_model
    return {
        "d_model": d_model,
        "bottleneck": bottleneck,
        "adapter_params": int(down + up),
        "down_proj": f"{d_model} -> {bottleneck}",
        "up_proj": f"{bottleneck} -> {d_model}",
    }
