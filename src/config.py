"""Central configuration for the dual-engine forecasting pipeline v3.

Changes in v3.1 (All-15-Fixes):
- ROAS_CAP lowered from 50 → 15 (Fix 14 — realistic ecommerce cap)
- 24 new FEATURE_COLS added: cyclical calendar, ecommerce events, budget ratio,
  interaction features, target encoding, decay lags, channel efficiency (Fix 5–10)
- OPTUNA_TRIALS = 50 for Bayesian hyperparameter search (Fix 12)
- LGBM_PARAMS updated with better defaults: shallower trees, more regularisation
- CHANNEL_WEIGHTS for imbalance correction (Fix 11)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HORIZONS = [30, 60, 90]
QUANTILES = [0.10, 0.50, 0.90]
QUANTILE_LABELS = ["p10", "p50", "p90"]

CHANNELS = ["google", "meta", "bing"]
CAMPAIGN_TYPE_ALLOWLIST = {
    "SEARCH",
    "PERFORMANCE_MAX",
    "DISPLAY",
    "VIDEO",
    "DEMAND_GEN",
    "SHOPPING",
    "REMARKETING",
    "PROSPECTING",
    "AUDIENCE",
    "BRAND",
    "UNKNOWN",
}

META_CONVERSION_AS_REVENUE = True

RANDOM_SEED = 42
HOLDOUT_WEEKS = 10
MC_SAMPLES = 10_000
MIN_TRAIN_SAMPLES = 10
BRAND_MIN_TRAIN = 8           # Action 5: Lower threshold for brand mini-model
BRAND_CAMPAIGN_TYPES = {"BRAND", "TM"}  # Action 5: Identifiers for brand campaigns

# Data sufficiency thresholds (weeks with revenue > 0)
DATA_THRESHOLD_FULL = 26      # ≥26 weeks → full LightGBM model
DATA_THRESHOLD_SPARSE = 12    # 12-25 weeks → LightGBM + wider bands (×1.25)

SPARSE_INTERVAL_MULTIPLIER = 1.25

# Groq API
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Optuna hyperparameter tuning (Fix 12) ───────────────────────────────────
OPTUNA_TRIALS = 50          # Number of Bayesian optimisation trials
OPTUNA_TIMEOUT = 600        # Max seconds per study (10 minutes safety cap)
OPTUNA_METRIC = "wape"      # Optimise WAPE (not SMAPE) — Fix 4

# ─── Channel imbalance weights (Fix 11) ──────────────────────────────────────
# Google dominates (~19k rows), Bing is tiny (~2.8k). Weights correct for this.
CHANNEL_WEIGHTS = {"google": 1.0, "meta": 3.0, "bing": 5.0}  # Phase 2: Meta 2.0 -> 3.0

# ─── Classifier gate thresholds (Fix 2) ──────────────────────────────────────
CLASSIFIER_THRESHOLD_P10 = 0.90  # Pessimistic bound — only show when very confident active
CLASSIFIER_THRESHOLD_P50 = 0.55  # Action 2: Median — moderate confidence (raised from 0.40)
CLASSIFIER_THRESHOLD_P90 = 0.10  # Optimistic bound — show even when uncertain

# ─── Q90 outlier cap percentile (Fix 3) ──────────────────────────────────────
Q90_CLIP_PERCENTILE = 95

# ─── LightGBM hyperparameters (updated defaults — Fix 12 guidance) ───────────
LGBM_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,     # Slightly higher (was 0.03) — faster convergence
    "num_leaves": 20,           # Shallower (was 31) — less overfitting per Optuna research
    "max_depth": 5,             # Explicit depth cap (new)
    "min_data_in_leaf": 15,     # More conservative (was 5) — reduce overfit
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l1": 0.1,
    "lambda_l2": 0.269,         # Increased (was 0.1) — per Optuna best practice
    "verbose": -1,
    "seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,          # Fix reproducibility: seeds row subsampling
    "feature_fraction_seed": RANDOM_SEED, # Fix reproducibility: seeds column subsampling
}

CATEGORICAL_FEATURES = ["channel", "campaign_type", "audience_segment"]

# ─── Unified Model Features ─────────────────────────────────────────────────
FEATURE_COLS = [
    # ── Planned Spend (dynamic scenario input) ──
    "planned_spend",
    "log_spend",

    # ── Budget ratio & interaction features (Fix 7) ──
    "budget_ratio",             # current spend / 4w rolling mean spend
    "log_proposed_spend",       # log1p(planned_spend)
    "spend_saturation",         # log1p(spend / median_channel_spend) — diminishing returns
    "spend_x_roas",             # spend_roll4 × roas_roll4 interaction
    "budget_ratio_x_roas",      # budget_ratio × roas_lag_1 interaction
    "log_spend_x_trend",        # log_proposed_spend × revenue_trend interaction
    "revenue_trend",            # roll_mean_4 / roll_mean_13 — short vs long momentum

    # ── Target encoding (Fix 8) ──
    "te_revenue",               # expanding mean revenue per (channel, campaign_type)
    "te_roas",                  # expanding mean ROAS per (channel, campaign_type)

    # ── Channel efficiency (Fix 10) ──
    "channel_avg_roas",         # channel-level average ROAS
    "spend_vs_channel_avg",     # campaign ROAS / channel avg ROAS

    # ── Global Momentum ──
    "global_portfolio_wow_change",

    # ── Revenue efficiency history ──
    "hist_roas_1",
    "hist_roas_roll4",
    "roas_rolling_mean_4",
    "roas_lag_1",
    "rpc_roll4",

    # ── Revenue lags ──
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_12",

    # ── Decay-weighted lags (Fix 9) ──
    "revenue_decay_4w",         # half-life=2 weeks exponentially weighted lag
    "spend_decay_4w",           # same for spend

    # ── YoY seasonal signal ──
    "lrev_lag_52",
    "lrev_lag_26",
    "yoy_ratio",
    "yoy_roll4_lag52",

    # ── Rolling revenue stats ──
    "roll_mean_4",
    "roll_mean_8",
    "roll_std_4",
    "roll_cv_4",
    "brand_roas_stability",
    "is_tm_brand",
    "brand_instability_signal",

    # ── Spend features (historical) ──
    "spend_lag_1",
    "spend_roll_4",
    "spend_ratio_vs_hist",

    # ── Cross-platform halo features ──
    "meta_spend_roll4",
    "meta_spend_roll8",
    "portfolio_upper_lower_ratio",
    "halo_lag_1w_google",
    "halo_lag_2w_google",
    "halo_lag_1w_meta",
    "halo_lag_2w_meta",
    "halo_lag_1w_bing",
    "halo_lag_2w_bing",

    # ── Calendar — raw (Fix 5 adds cyclical versions) ──
    "week_of_year",
    "month",
    "quarter",
    "year",

    # ── Calendar — cyclical encoding (Fix 5) ──
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",

    # ── Ecommerce event flags (Fix 6) ──
    "is_q4",
    "is_holiday_adj",
    "is_black_friday_week",
    "is_cyber_week",
    "is_christmas_week",
    "is_back_to_school",
    "is_jan_slump",
    "is_valentines_week",

    # ── Campaign maturity ──
    "weeks_active",
    "is_new_campaign",
    "days_since_start",
    "campaign_age_weeks",

    # ── Zero-inflation ──
    "recent_zero_rate",
    "last_week_zero",
    "consecutive_zeros",

    # ── Categoricals ──
    "channel",
    "campaign_type",
    "audience_segment",
]

# ─── ROAS & evaluation thresholds ────────────────────────────────────────────
ROAS_EVAL_MIN_SPEND = 5.0
ROAS_CAP = 15.0              # Fix 14: Lowered from 50 → 15 (realistic ecommerce max)
INTERVAL_TARGET_COVERAGE = 80.0
CONFORMAL_TARGET_COVERAGE = 0.80  # Action 1: Target coverage for per-group conformal calibration

# Curve fitting
CURVE_MIN_POINTS = 20

# Budget simulation safety bounds
BUDGET_RATIO_MIN = 0.1
BUDGET_RATIO_MAX = 5.0
