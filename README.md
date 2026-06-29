# 🛡️ Banking Fraud Detection & AML — LSTM Sequence Engine

Production-ready deep-learning project for transaction-sequence fraud / AML
detection on the **PaySim** dataset.

```
PaySim sequence (N_TIMESTEPS=5, N_FEATURES=8)
  → Conv1D(64, k3, s1) → Conv1D(32, k3, s2)
  → LSTM(64) → LSTM(32)
  → Multi-Head Self-Attention (4 heads, key_dim 16) + residual + LayerNorm
  → Dense(32) latent
  → XGBoost stacking meta-learner  (latent 32 ⊕ raw tabular 8)
  → Fraud probability → CLEAR (<0.30) / REVIEW (0.30–0.70) / FREEZE (>0.70)
```

**Error ceiling:** ROC-AUC ≥ 0.97, Recall ≥ 0.97, F1 ≥ 0.97 (error ≤ 3%).

## Run locally

```bash
pip install -r requirements.txt
# place the dataset at ./data/paysim.csv  (or keep it in the project root)
streamlit run app.py
python generate_slides.py        # -> Banking_Fraud_AML_Deck.pptx
```

## Run with Docker

```bash
docker-compose up --build        # -> http://localhost:8501
```

The `./data` folder is mounted as a volume; set `PAYSIM_CSV` to override the path.

## Layout

| Path | Purpose |
|------|---------|
| `app.py` / `app_tabs.py` | 6-tab Streamlit dashboard |
| `models/architecture.py` | Conv1D + LSTM + Attention (Keras) |
| `models/train.py` | data prep, class weights, early stopping |
| `models/ensemble.py` | XGBoost stacking on LSTM latent features |
| `models/adapters.py` | transfer-learning bottleneck adapters |
| `models/rnn_baseline.py`, `models/xgb_baseline.py` | battle competitors |
| `models/evaluate.py` | rule-based model, routing, benchmarks |
| `utils/` | features, sequences, metrics, keywords, SAR, sanctions |
| `generate_slides.py` | 12-slide python-pptx deck |

## Dashboard tabs

1. **Training Data** — PaySim EDA (class balance, correlations, sequence preview)
2. **Model Architecture** — visual + math for Conv1D / LSTM / Attention / Adapters / XGBoost / NLP
3. **Fine-Tuning** — hyperparameter grid, training curves, early stopping, error monitor
4. **Model Battle** — Rule vs XGBoost vs Simple RNN vs full ensemble
5. **Continuous Training & MCP** — drift simulator, transaction scorer, sanctions screening
6. **Advanced Maths & CS** — vanishing gradient, BPTT, weighted CE, ROC-AUC, complexity, DSA
