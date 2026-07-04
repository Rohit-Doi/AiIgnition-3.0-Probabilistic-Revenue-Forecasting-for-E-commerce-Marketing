# Architecture Overview

## Stack

| Layer | Technology |
|---|---|
| Forecasting core | LightGBM quantile regression, scipy curve fitting, Monte Carlo sampling |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| LLM | Groq (llama-3.3-70b-versatile) |
| Data | Pandas, PyArrow (Parquet features) |

## Pipeline Flow

```
data/*.csv
    │
    ▼
load_data.py ──► validate.py ──► aggregate.py (weekly)
    │
    ▼
generate_features.py ──► features.parquet
    │
    ▼
predict.py + model.pkl ──► recursive weekly rollout (MC) ──► output/predictions.csv
```

## Scoring vs. Interactive Paths

```
┌─────────────────────────────────────────────────┐
│  run.sh (OFFLINE — no network)                  │
│  generate_features → predict → predictions.csv    │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  app.py / api.py (INTERACTIVE)                  │
│  forecast → simulate → insights (Groq API)       │
└─────────────────────────────────────────────────┘
```

## Key Modules

| Module | Responsibility |
|---|---|
| `src/load_data.py` | Multi-channel CSV harmonization & anomaly flagging |
| `src/validate.py` | Data quality checks + cleaning |
| `src/aggregate.py` | Weekly aggregation + feature engineering + data tiers |
| `src/model.py` | LightGBM log-scale training, fallback models |
| `src/curve.py` | Spend-response curve competition & safety bounds |
| `src/reconcile.py` | ROAS computation and channel reconciliation |
| `src/pipeline.py` | End-to-end forecast generation with Monte Carlo rollout |
| `src/insights.py` | 4 distinct Groq LLM integrations + SHAP importance |
| `src/metrics.py` | SMAPE, WMAPE, WSMAPE, pinball loss, coverage |
| `src/train.py` | Offline model training & validation |
| `src/api.py` | REST API endpoints |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/validate` | Data validation report |
| POST | `/forecast` | Generate probabilistic forecasts |
| POST | `/simulate` | Budget scenario simulation |
| POST | `/insights` | AI-assisted business insights (4 functions) |
| GET | `/metrics` | Backtest calibration metrics (w/ calibration curve) |

## Deployment Notes

- Model artifact: `pickle/model.pkl` (pre-trained, committed)
- Python 3.11+, all deps pinned in `requirements.txt`
- Set `GROQ_API_KEY` for AI insights (optional)
- No internet connection required to run `run.sh`
