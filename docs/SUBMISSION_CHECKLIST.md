# Submission Checklist — AIgnition 3.0

Track progress against hackathon submission guide and project plan.

## Submission Guide (Required)

- [ ] Repo on GitHub, set to **public**
- [x] `run.sh` at root, accepts `DATA_DIR`, `MODEL_PATH`, `OUTPUT_PATH` with defaults
- [x] `run.sh` runs end-to-end: features → predict → output (no retrain, no network)
- [x] `data/` folder exists; code reads CSVs dynamically by pattern
- [x] Trained model committed at `pickle/model.pkl`
- [x] `requirements.txt` with **pinned** versions
- [x] Output written to `OUTPUT_PATH` as `predictions.csv`
- [ ] Full sequence tested on **fresh clone** in clean environment
- [x] No absolute paths, no prompts, no internet at scoring time
- [ ] Team name, members, college added to README
- [ ] Submit GitHub URL + command to sunitha.k@netelixir.us by **July 19, 2026 10:00 PM IST**

## Deliverables (Project Brief)

- [x] Working prototype: ingest, validate, forecast, simulate, AI insights
- [x] Technical documentation (`docs/methodology.md`)
- [x] Architecture overview (`docs/architecture.md`)
- [ ] Demo walkthrough / presentation recording

## Forecasting Plan

- [x] Probabilistic P10/P50/P90 revenue forecasts
- [x] Derived ROAS ranges (revenue ÷ spend scenario)
- [x] 30 / 60 / 90-day horizons
- [x] Channel / campaign-type / campaign levels
- [x] Weekly aggregation (aggregate-period, not daily)
- [x] LightGBM quantile regression
- [x] Spend-response curves with per-type form selection
- [x] Monte Carlo quantile reconciliation
- [x] Meta conversion-field verification documented
- [x] Daily vs weekly ablation (`docs/ablation_results.csv`)
- [x] Backtest metrics: SMAPE, pinball loss, coverage
- [x] LLM insights (Groq llama-3.3-70b-versatile) — outside scored pipeline
- [x] Anomaly detection via P10–P90 band breaches
- [x] Streamlit dashboard + FastAPI backend

## Before You Submit

```bash
git clone <your-repo>
cd <repo>
pip install -r requirements.txt
pytest tests/
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
streamlit run app.py   # optional demo
```

Expected output: `output/predictions.csv` with 100+ rows across horizons and levels.
