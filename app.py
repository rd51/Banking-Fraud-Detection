"""
Banking Fraud Detection & AML — Streamlit dashboard.

    streamlit run app.py

Six tabs:
    1. Training Data            — PaySim EDA
    2. Model Architecture       — visual + math (Conv1D / LSTM / Attention / Adapter / XGBoost / NLP)
    3. Fine-Tuning              — hyperparameter grid + error ceiling monitor
    4. Model Battle             — 4-model comparison + error ceiling monitor
    5. Continuous Training & MCP — drift sim, transaction scorer, sanctions
    6. Advanced Mathematics & CS — equations + data structures

Everything heavy is wrapped in @st.cache_resource / @st.cache_data.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import branding as B
from utils.features import FEATURES, TARGET, load_paysim, find_csv, engineer
from utils.sequences import build_sequences, pad_single_sequence, sliding_window_pseudocode
from utils.metrics import (
    evaluate, confusion, roc_points, pr_points, best_threshold,
    weighted_cross_entropy, CEILING, MAX_ERROR,
)
from utils.keywords import extract_entities, entity_tags, extractor_logic_source
from utils.sar_draft import generate_sar
from utils.sanctions import screen, MOCK_OFAC
from models.evaluate import rule_based_scores, decision_route, benchmark_inference

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Banking Fraud & AML — LSTM Sequence Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(B.GLOBAL_CSS, unsafe_allow_html=True)


def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


HAS_TF = tf_available()


# --------------------------------------------------------------------------- #
# Cached data + model loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading PaySim data…")
def load_eda(nrows: int, uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Engineered EDA frame (sampled). Honours an uploaded file when provided."""
    if uploaded_bytes is not None:
        import io
        raw = pd.read_csv(io.BytesIO(uploaded_bytes), nrows=nrows)
        return engineer(raw)
    return load_paysim(nrows=nrows, balance=False)


@st.cache_resource(show_spinner="Preparing balanced training windows (capturing all fraud)…")
def get_dataset(n_timesteps: int, legit_per_fraud: int):
    """Full-file balanced dataset — reads the whole file so all ~8,213 fraud
    cases are captured (the NROWS slider only affects the EDA preview)."""
    from models.train import prepare_dataset
    return prepare_dataset(n_timesteps=n_timesteps, legit_per_fraud=legit_per_fraud)


@st.cache_resource(show_spinner="Training LSTM RNN…")
def train_primary(n_timesteps: int, legit_per_fraud: int,
                  lstm_units: int, dropout: float, learning_rate: float,
                  epochs: int):
    """Train the BRD LSTM RNN; cached on every hyperparameter combination."""
    from models.train import train_lstm
    ds = get_dataset(n_timesteps, legit_per_fraud)
    return train_lstm(
        ds, lstm_units=lstm_units, dropout=dropout, learning_rate=learning_rate,
        epochs=epochs,
    )


@st.cache_resource(show_spinner="Pre-training base LSTM then fine-tuning…")
def train_pretrain(n_timesteps: int, legit_per_fraud: int,
                   lstm_units: int, dropout: float, learning_rate: float,
                   pretrain_epochs: int, finetune_epochs: int):
    """Run the pretrain → fine-tune transfer-learning demonstration (cached)."""
    from models.train import train_with_pretraining
    ds = get_dataset(n_timesteps, legit_per_fraud)
    return train_with_pretraining(
        ds, lstm_units=lstm_units, dropout=dropout, learning_rate=learning_rate,
        pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
    )


# --------------------------------------------------------------------------- #
# Sidebar controls (shared)
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🛡️ AML Control Panel")
csv_path = find_csv()
if csv_path:
    st.sidebar.success(f"Dataset: `{csv_path}`")
    uploaded = None
else:
    st.sidebar.error("paysim.csv not found.")
    up = st.sidebar.file_uploader("Upload paysim.csv", type=["csv"])
    uploaded = up.getvalue() if up is not None else None

st.sidebar.markdown("### Data scale")
NROWS = st.sidebar.select_slider(
    "Rows for EDA preview (Training Data tab)",
    options=[100_000, 300_000, 500_000, 800_000, 1_200_000, 2_000_000],
    value=500_000,
    help="Only the Training Data charts use this. Model training always reads "
         "the full file so every fraud case is captured.",
)
LEGIT_PER_FRAUD = st.sidebar.slider("Legit : fraud ratio (training)", 4, 30, 12)
EPOCHS = st.sidebar.slider("Max epochs (LSTM)", 5, 40, 15)

st.sidebar.markdown("---")
st.sidebar.caption("SP Jain MAIB · Deep Learning Group Project · June 2026")
st.sidebar.caption(f"TensorFlow available: {'✅' if HAS_TF else '❌ (deep tabs limited)'}")


def error_monitor(error: float, where: str = ""):
    """Render the shared red/green error-ceiling monitor."""
    within = error <= MAX_ERROR
    col1, col2 = st.columns([1, 2])
    col1.metric(
        f"Error rate {where}".strip(),
        f"{error*100:.2f}%",
        delta=f"{(MAX_ERROR-error)*100:+.2f}% vs ceiling",
        delta_color="normal" if within else "inverse",
    )
    if within:
        col2.markdown(
            "<div class='badge-clear'>🟢 Within Ceiling — error ≤ 3%</div>",
            unsafe_allow_html=True)
    else:
        col2.markdown(
            "<div class='badge-freeze blink'>🔴 EXCEEDS 3% LIMIT — REJECT</div>",
            unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header band
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class='brand-band'>
      <h1>🛡️ Banking Fraud Detection &amp; AML</h1>
      <p>Conv1D → LSTM → Multi-Head Attention → XGBoost Stacking Ensemble ·
         From a USD 3.09 B fine to real-time sequence detection</p>
    </div>
    """,
    unsafe_allow_html=True,
)

TABS = st.tabs([
    "📊  Training Data",
    "🧠  Model Architecture · LSTM RNN",
    "🎛️  Fine-Tuning",
    "🔁  Continuous Training & MCP",
])

# Import tab renderers (kept in the same file below).
from app_tabs import (  # noqa: E402
    render_tab_data, render_tab_architecture, render_tab_finetune,
    render_tab_mcp,
)

ctx = dict(
    NROWS=NROWS, LEGIT_PER_FRAUD=LEGIT_PER_FRAUD, EPOCHS=EPOCHS,
    uploaded=uploaded, HAS_TF=HAS_TF,
    load_eda=load_eda, get_dataset=get_dataset,
    train_primary=train_primary, train_pretrain=train_pretrain,
    error_monitor=error_monitor,
)

with TABS[0]:
    render_tab_data(ctx)
with TABS[1]:
    render_tab_architecture(ctx)
with TABS[2]:
    render_tab_finetune(ctx)
with TABS[3]:
    render_tab_mcp(ctx)
