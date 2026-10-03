"""Central configuration for the AIgnition forecasting engine (v4)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42

# ─── Forecast contract ───────────────────────────────────────────────────────
HORIZONS = [30, 60, 90]            # planning windows, in days after the forecast origin
INTERVAL_COVERAGE = 0.80           # P10–P90 band

CAMPAIGN_TYPE_ALLOWLIST = {
    "SEARCH", "PERFORMANCE_MAX", "DISPLAY", "VIDEO", "DEMAND_GEN", "SHOPPING",
    "REMARKETING", "PROSPECTING", "BRAND", "UNKNOWN",
}

# Meta exports a `conversion` column whose values behave like purchase value, not
# like a conversion count (see docs/methodology.md §1). Set False to drop Meta
# revenue from every forecast instead.
META_CONVERSION_AS_REVENUE = True

# ─── Data maturity ───────────────────────────────────────────────────────────
# Ad platforms keep attributing conversions to a click for days afterwards, so the
# newest days of any extract under-report revenue. Revenue for this many days
# before the forecast origin is nowcast from spend x trailing ROAS instead.
MATURITY_LAG_DAYS = 2
NOWCAST_ROAS_CAP = 30.0            # ceiling on the trailing ROAS used by the nowcast

# ─── Seasonality ─────────────────────────────────────────────────────────────
# Seasonal analogs: the same window in other years, shifted by whole weeks so
# weekdays line up. Positive = earlier years, negative = later years (the latter
# can only ever be valid for training samples).
SEASON_SHIFT_DAYS = 364
SEASON_YEARS = [1, 2, -1, -2]
# Each analog year is read at these offsets (days) and averaged: promotions and
# peak weeks do not land on the same calendar week every year.
SEASON_OFFSET_DAYS = (0,)
SEASONAL_CLIP = (-2.0, 2.5)        # bounds on an analog log-ratio (x0.14 … x12)

# Peak retail season (inclusive month/day bounds). Windows that touch it get
# their own seasonal coefficient because the analog ratio is signal there and
# mostly promo noise elsewhere.
PEAK_SEASON = ((11, 18), (12, 24))

# ─── Stage 1: structural log-linear median ───────────────────────────────────
RIDGE_ALPHA = 10.0                 # L2 penalty (sample weights are normalised to mean 1)
# A 7- or 14-day run-rate within this log-distance of the 28-day run-rate (about
# +/-35%) is ordinary week-to-week noise and is ignored; only the excess counts.
MOMENTUM_DEAD_ZONE = 0.35
SAMPLE_WEIGHT_POWER = 0.5          # weight ∝ (series revenue run-rate) ** power
TRAIN_ORIGIN_STEP_DAYS = 7         # spacing of training forecast origins

# ─── Stage 2: quantile boosting on stage-1 residuals ─────────────────────────
QGBM_ROUNDS = 150
QGBM_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}

# ─── Stage 3: conformal calibration ──────────────────────────────────────────
CONFORMAL_MIN_SAMPLES = 40         # below this a bucket borrows from the pooled horizon
CONFORMAL_MIN_ORIGINS = 8          # …and it must span at least this many forecast origins
CONFORMAL_DEFAULT_OFFSET = 0.25    # log-units, used only when nothing can be calibrated

# ─── Rolling-origin backtest ─────────────────────────────────────────────────
BACKTEST_MIN_TRAIN_DAYS = 500      # first test origin needs this much history
BACKTEST_STEP_DAYS = 7             # spacing of test origins

# ─── Budget response ─────────────────────────────────────────────────────────
ELASTICITY_BOUNDS = (0.40, 1.00)   # revenue pass-through of a spend change
BUDGET_MULTIPLIER_BOUNDS = (0.25, 3.0)   # scenario range offered by the simulator
EXTRAPOLATION_WARN_RATIO = 1.5     # plan vs. largest historical spend for that window

# ─── Reporting ───────────────────────────────────────────────────────────────
ROAS_MIN_SPEND = 1.0               # below this much horizon spend, ROAS is reported as 0
ROAS_CAP = 100.0                   # numerical guard only; forecasts are not clipped to a "realistic" ROAS

# ─── LLM layer (dashboard / API only — never used by run.sh) ─────────────────
# Any OpenAI-compatible chat endpoint works. Providers are tried in this order;
# for each one, every configured key (NAME, NAME_FALLBACK, NAME_2) is tried, and
# the first model in `models` that the key is allowed to use is selected, because
# hosted model line-ups change. Set LLM_MODEL to force a specific model.
LLM_PROVIDERS = {
    "groq": {
        "env": ["GROQ_API_KEY"],
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "models_url": "https://api.groq.com/openai/v1/models",
        "models": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
    },
    "gemini": {
        "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/openai/models",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash"],
    },
    "openai": {
        "env": ["OPENAI_API_KEY"],
        "url": "https://api.openai.com/v1/chat/completions",
        "models_url": "https://api.openai.com/v1/models",
        "models": ["gpt-4o-mini"],
    },
}
LLM_TIMEOUT_SECONDS = 45
