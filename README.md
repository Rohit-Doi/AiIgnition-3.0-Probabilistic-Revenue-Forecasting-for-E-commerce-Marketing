# AIgnition 3.0 — Probabilistic Revenue Forecasting for E-commerce Marketing
> *An AI-Assisted Forecasting Utility for Digital Marketing Agencies*

**Team:** The Trident — Shiva Krishna Sherikar (Team Lead), Kamatam Rohit, C. Namish  
**College:** Malla Reddy College of Engineering and Technology
**Hackathon:** AIgnition 3.0 (by Netelixir) 

## 📖 Introduction
E-commerce businesses invest across multiple acquisition channels such as Google Ads, Meta Ads, Microsoft Ads, organic search, affiliate networks, and display campaigns. Digital marketing agencies managing these businesses are expected to estimate future business outcomes before budgets are deployed. This is a difficult forecasting problem. 

Marketing performance is influenced by seasonality, changing user behavior, inconsistent campaign structures, varying channel efficiency, and incomplete analytics visibility. Existing forecasting workflows are often spreadsheet-driven, manually maintained, and disconnected across platforms. 

This hackathon challenges participants to build a practical AI-assisted forecasting utility that predicts ecommerce revenue and ROAS using historical analytics and sales data. 

**Our Solution:** AIgnition focuses on realistic business forecasting rather than theoretical modeling, clearly communicating assumptions, uncertainty, and operational insights through a **Direct Horizon Residual Boosting architecture with Binary Zero-Inflation Gating**, spend-response curves, Monte Carlo simulations, and Groq-powered intelligence.

---

## 🏆 Why Our Solution is Different — Key Contributions

| # | Innovation | What Makes It Different |
|---|---|---|
| 1 | **Direct Horizon Residual Boosting** | Trains one model per horizon (30/60/90 d) — eliminates recursive error compounding that cripples standard time-series stacks |
| 2 | **MACD Statistical Anchoring** | Revenue residuals are predicted *against* a momentum anchor (fast/slow EMA ratio), making the boosting task far more bounded and stable |
| 3 | **Binary Zero-Inflation Gating** | A LightGBM classifier gates every forecast with asymmetric probability thresholds, handling the zero-revenue campaigns that break standard regressors |
| 4 | **Saturation-Aware Budget Simulation** | Hill/S-curve saturation fitted per campaign type — "What-If" sliders show realistic diminishing returns, not naive linear scaling |
| 5 | **Groq CMO Layer** | LLM receives marginal ROAS derivatives + SHAP-like importances — strategic recommendations, not just raw data summaries |
| 6 | **Verifiable No-Leakage Guarantee** | `assert max_train_time < min_val_time` crashes the pipeline if the temporal split is ever violated — not just a claim |

---

## 🌟 Unique Dashboard Features & Visualizations

We built AIgnition 3.0 to be a highly operational tool for marketing agencies. To stand out, we went beyond standard tables and implemented advanced, interactive visualizations that directly answer the hackathon's prompt to "use additional functions to forecast revenue based on different media budgets":

### 1. The Budget Simulator & Spend-Response Curves
Instead of scaling revenue linearly (which ignores diminishing returns), the dashboard features a dedicated **Budget Simulator**. 
* **Hill Curve Saturation:** The model fits mathematical S-curves (Hill functions) to historical data for every campaign type to find the exact "K-value" saturation point (where spending more money stops generating proportional revenue).
* **Interactive What-Ifs:** Adjusting the budget slider dynamically passes new spend scenarios into these curves, showing realistic P10/P50/P90 revenue impacts alongside the status-quo forecast.

### 2. The Efficient Frontier Scatter Plot
Also located in the Budget Simulator tab, this graph visualizes simulated revenue vs. spend for every campaign type, with bubble size representing ROAS. This instantly tells an agency exactly where their next marginal dollar of budget will work the hardest.

### 3. Portfolio Money Flow (Sankey Diagram)
A dynamic flow chart that visually maps the pipeline from **Total Budget → Channel Allocation → Projected Revenue**. The thickness of the bands instantly communicates which channels are highly efficient (thick revenue band, thin spend band) and which are burning cash.

### 4. Groq-Powered "Chat With Your Forecast"
Integrated a Llama 3.3 70B reasoning layer (via Groq) that doesn't just read raw data, but interprets the model's marginal ROAS derivatives and feature importances. Users can type questions like *"If I move $5k from Meta to Google Search, what happens?"* and get strategic, CMO-level answers based on the current scenario.

---

## 🚀 Quick Start

### Requirements
- **Python 3.11+**
- Dependencies rigidly pinned in `requirements.txt`
- No internet connection required during offline scoring execution.

### Optional: AI Insights Configuration (Groq)
The Streamlit dashboard features an "AI Insights" tab that generates strategic explanations of the forecasts using Groq's Llama 3.3 70B model. To enable this:
1. Get a free API key from **[console.groq.com](https://console.groq.com/keys)**
2. Rename `.env.example` to `.env`
3. Paste your key: `GROQ_API_KEY=your_key_here`

*(Note: The `run.sh` offline scoring script does not use this key or make any network calls. It is strictly for the frontend dashboard.)*

### Execution

**Note for Windows Users:** The scoring pipeline is a bash script (`run.sh`). You must run this using a bash-compatible terminal like **Git Bash** (not PowerShell or Command Prompt). 

> **Environment Setup Note:** If you run `pip install -r requirements.txt` globally, `run.sh` will auto-detect your Python environment. You do *not* need to manually activate a `.venv` for the scoring script to work. It will automatically find the Python installation that contains the required packages.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (one-time; trained model is committed for scoring)
python -m src.train

# 2a. With Optuna hyperparameter tuning (50 trials, ~10 min extra)
python -m src.train --optuna

# 2b. With 3-fold walk-forward cross-validation output
python -m src.train --cv

# 3. Run offline scoring pipeline (submission entry point)
# Run this exactly as written in Git Bash (or Linux/Mac terminal)
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv

# 4. Launch interactive dashboard
streamlit run app.py

# 5. (Optional) Launch API
uvicorn src.api:app --reload --port 8000
```

---

---

## 🧠 How It Works — Architecture in Brief

![AIgnition Pipeline Architecture](docs/pipeline_architecture.png)

AIgnition uses a **Dual-Engine, Direct Horizon Residual Boosting** system. The four core ideas:

1. **MACD Anchoring + Residual Boosting** — Instead of predicting raw revenue (extremely volatile), LightGBM learns the *residual* between a momentum anchor (fast/slow EMA ratio) and the actual forward-summed revenue in log-space. The anchor handles bulk signal; the model handles fine corrections.
2. **Binary Zero-Inflation Gate** — A `LGBMClassifier` per group predicts `P(revenue > 0)` with asymmetric thresholds before the quantile regressor runs. This drove coverage from 62.9% → 85.7%.
3. **Direct Horizon Models** — Separate models trained for 30 / 60 / 90-day targets eliminate recursive error compounding entirely.
4. **Conformal Calibration** — A post-training empirical multiplier ensures P10–P90 intervals hit the 80% coverage target on the validation fold.

> Full data assumptions, feature engineering (55+ features), Optuna tuning details, and the complete ablation study are in [`docs/methodology.md`](docs/methodology.md).

---

## 📐 Mathematical Metric Definitions

> Full formulae, derivations, and metric rationale are in [`docs/methodology.md`](docs/methodology.md#-mathematical-metric-definitions). Below is a brief reference:
> **WAPE** (primary) = Σ|y − ŷ| / Σy · 100 — dollar-weighted error that correctly penalises large-revenue misses.
> **Coverage** = fraction of actuals falling inside the P10–P90 band (target ≥ 80%).

---

## 🔬 What This Solution Addresses

The brief asks for: probabilistic forecasting, multi-horizon outputs, AI-assisted interpretation, and operational usefulness for agencies. Here is how each is specifically implemented — not as a feature claim but as a description of the actual code:

| Brief Requirement | How It's Implemented |
|---|---|
| Probabilistic revenue forecast | Calibrated P10/P50/P90 quantile models with conformal calibration; 85.7% empirical coverage vs 80% target |
| Multi-horizon (30/60/90 day) | Direct horizon models trained per horizon — no recursive compounding of errors |
| Handles zero-inflation | LGBMClassifier binary gate per group with asymmetric thresholds (P10: prob>0.90, P90: prob>0.10) |
| No data leakage | Programmatic `assert max_train_time < min_val_time` in `model.py` — crashes the pipeline if violated |
| Reproducible results | All LightGBM random seeds locked; identical output on every run from the same data |
| Spend-response simulation | Hill/Saturating-exponential/Log/Linear curves fitted per campaign type; saturation heatmap in dashboard |
| AI-generated insights | LLM receives marginal ROAS derivatives, SHAP-like importances, and spend anomaly signals — not just raw data |
| Agency-ready output | One-click CSV export of forecast scenario with ROAS breakdowns and interval bounds |
| Validated against baselines | Beat lag_1 by +64.2 pp WAPE, rolling_mean_4 by +81.0 pp, seasonal_lag by +144.2 pp |

---

## 📊 Performance Against the Baseline

### 📋 At-a-Glance Summary

| Metric | Value | Notes |
|---|---|---|
| **WAPE (channel-level)** | **35.8%** | Aggregated channel × campaign-type (70 rows) — planning resolution |
| **WAPE (campaign-level)** | 89.0% | Per-campaign (414 rows) — directional signal, not precision |
| **P10–P90 Coverage** | **85.7%** | Exceeds 80% target; conformal-calibrated |
| **Beat lag_1 baseline by** | **+64.2 pp** | Standard hardest-to-beat time-series baseline |
| **Forecast Horizons** | 30 / 60 / 90 days | Direct (non-recursive) multi-horizon models |
| **AI Features** | ROAS derivatives, SHAP importances, spend-response curves | Fed to Groq Llama 3.3 70B for strategic insights |

---

**Primary Metric: WAPE (Weighted Absolute Percentage Error)** — weights errors by actual revenue size, so large-revenue weeks dominate, which is the correct metric for ecommerce.

### Headline Results — Two Resolutions, Reported Honestly

This model produces forecasts at two resolutions. We report both as primary, because the brief asks for both, and they answer different questions.

| Resolution | WAPE | What it tells you |
|---|---|---|
| Campaign-level (414 rows) | 89.0% | Can we predict any single campaign's revenue next month? Directionally, yes. |
| Channel×Type-level (70 rows, aggregated) | **35.8%** | Can we predict a channel's total weekly revenue for budget planning? Yes, reliably. |

**Why both numbers exist and neither is wrong:** individual ecommerce campaigns are zero-inflated and spike unpredictably week to week — a $300/week remarketing campaign can go to $2,000 or $0 with no warning, and no model trained on 10 weeks of holdout data will catch that reliably. When you sum many such campaigns into a channel total, the campaign-level noise partially cancels (this is a real statistical property, not a trick), which is why the channel-level number is so much better.

**What this means for the product**: campaign-level outputs in this prototype should be read as directional, not precise. Channel and campaign-type forecasts are the resolution we'd recommend an agency actually budget against.

### Baseline Comparison

We compare against three naive baselines. `lag_1` (predict next period = last period) is the standard, hardest-to-beat baseline in time series forecasting — we report improvement against it as the headline number.

| Baseline | WAPE | Our model | Improvement |
|---|---|---|---|
| **lag_1 (standard baseline)** | **100.0%** | **35.8%** | **+64.2 pp** |
| rolling_mean_4 | 116.8% | 35.8% | +81.0 pp |
| seasonal_lag4_12 | 180.0% | 35.8% | +144.2 pp |

We beat all three. We lead with lag_1 because it's the conventional benchmark in the forecasting literature; the larger improvement numbers against weaker baselines are included for completeness, not because they're more impressive.

---

---



## 🤖 Strategic AI Insights (Groq Integration)

The LLM does **not** read raw data. Our Python pipeline computes and feeds it: marginal ROAS derivatives, spend-response saturation curves, and SHAP-like feature importances. This grounds every recommendation in the model's actual math.

**4 Distinct AI Functions:**
1. **Forecast Explanation** — explains *why* the P50 revenue is shifting
2. **Anomaly Interpretation** — investigates sudden drops or spikes in the holdout
3. **Budget Recommendation** — recommends cross-channel shifts based on curve saturation
4. **Portfolio Insight** — summarises the overall health of the multi-channel marketing mix

> The Groq API is strictly isolated to the frontend. The `run.sh` scoring pipeline is fully offline — no network calls, satisfying the hackathon's offline requirement.



## 📂 Project Structure (The Product)

You are reviewing a full ad-tech intelligence product, organized as follows:

```
/AIgnition
├── app.py                 # Streamlit Dashboard (Frontend) — root level
├── run.sh                 # Fully offline, end-to-end scoring script (bash)
├── run.bat                # Windows equivalent of run.sh
├── requirements.txt       # All dependencies, version-pinned
├── README.md              # Submission documentation
├── data/                  # Source CSVs (Google, Bing, Meta)
├── output/                # Final predictions.csv (generated by run.sh)
├── pickle/                # Serialized model bundles (model.pkl)
├── docs/                  # Extended documentation
│   ├── architecture.md    # System design & API endpoint reference
│   ├── methodology.md     # Complete mathematical methodology
│   ├── results.md         # Holdout backtest analysis
│   └── validation_results.json  # Raw metric dump
├── src/                   # Core Engine
│   ├── aggregate.py       # Feature engineering & temporal aggregation
│   ├── api.py             # FastAPI implementation
│   ├── config.py          # Global variables & hyperparameter tuning
│   ├── curve.py           # Spend-response curve fitting
│   ├── generate_features.py  # Feature generation entry point (for run.sh)
│   ├── insights.py        # Groq LLM integration
│   ├── load_data.py       # Data ingestion & harmonization
│   ├── metrics.py         # Backtest & evaluation logic
│   ├── model.py           # Dual-engine LightGBM logic
│   ├── pipeline.py        # Forecast generation logic
│   ├── predict.py         # Prediction entry point (for run.sh)
│   ├── reconcile.py       # Monte Carlo interval summation
│   ├── train.py           # Training and evaluation entry point
│   └── validate.py        # Data sanity checks
└── tests/                 # Automated test suite
    └── test_pipeline.py   # ROAS cap, clipping, and monotonicity tests
```

### Output Formatting Contract

`predictions.csv` precisely matches the hackathon requirement format:

| Column | Description |
|---|---|
| `channel` | google / meta / bing / all |
| `campaign_type` | SEARCH, PERFORMANCE_MAX, etc. |
| `campaign_name` | Campaign identifier |
| `horizon_days` | 30, 60, or 90 |
| `p10_revenue` | Pessimistic revenue forecast |
| `p50_revenue` | Most likely revenue forecast |
| `p90_revenue` | Optimistic revenue forecast |
| `p10_roas` | Derived pessimistic ROAS |
| `p50_roas` | Derived most likely ROAS |
| `p90_roas` | Derived optimistic ROAS |

---

## 📚 For More Information

To explore the exact mathematics, metrics, and architecture in even deeper detail, please refer to the dedicated documentation files located in the `docs/` folder:

- **`docs/architecture.md`**: Complete system design, pipeline flow, and API endpoint documentation.
- **`docs/methodology.md`**: Exhaustive breakdown of data assumptions, feature engineering (including top 10 features), sample weighting, and time-series contiguous filling.
- **`docs/results.md`**: In-depth analysis of the holdout backtest, addressing the recursive error snowball and outlier behavior.
- **`docs/validation_results.json`**: The raw computational dump of all calibration metrics (including pinball loss, WMAPE, RMSE, and specific outlier diagnostics).
