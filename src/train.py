"""Train quantile models, backtest on holdout, compare vs baselines.

v3.1 Changes (All-15-Fixes):
- Fix 4: WAPE is now the PRIMARY printed metric (was SMAPE)
- --optuna flag: triggers Optuna 50-trial hyperparameter tuning (Fix 12)
- --cv flag: triggers 3-fold walk-forward cross-validation (Fix 13)
- ROAS cap updated to 15x throughout (Fix 14)
- Adds wape_p50 to holdout metrics and all print outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import HOLDOUT_WEEKS, META_CONVERSION_AS_REVENUE, ROOT
from src.metrics import compare_to_baselines, evaluate_predictions, metrics_table
from src.config import ROAS_CAP, ROAS_EVAL_MIN_SPEND
from src.model import ModelBundle, model_key, predict_group, save_bundle, train_quantile_models
from src.pipeline import prepare_data


def run_backtest(bundle: ModelBundle, test_df: pd.DataFrame) -> dict:
    """Evaluate on unseen holdout weeks: WAPE (primary) + SMAPE + MAE + coverage + baselines."""
    y_true, p10, p50, p90, spend = [], [], [], [], []
    by_type_records = []

    nonzero_revenue = test_df.loc[test_df["revenue"] > 0, "revenue"]
    revenue_min_quantile = float(np.quantile(nonzero_revenue, 0.10)) if len(nonzero_revenue) else 0.0
    revenue_min = max(revenue_min_quantile, 500.0)
    spend_min = ROAS_EVAL_MIN_SPEND

    # Exclude zero-fill rows (added by fill_missing_weeks) from backtest
    test_df = test_df[~((test_df["spend"] == 0) & (test_df["revenue"] == 0))].copy()

    # Exclude rows without future targets
    test_df = test_df[test_df["target_30"].notna()].copy()
    test_df = test_df[test_df["target_30"] > 0].copy()

    for key in test_df["model_key"].unique() if "model_key" in test_df.columns else []:
        subset = test_df[test_df["model_key"] == key]
        ct = key.split("|", 1)[1]
        yt, pt10, pt50, pt90, sp = [], [], [], [], []
        for _, row in subset.iterrows():
            target = float(row["target_30"])
            preds = predict_group(bundle, row, horizon=30, campaign_type=ct)
            # Scale down to weekly run-rate for apples-to-apples comparison
            yt.append(target / 4.0)
            pt10.append(preds["p10"] / 4.0)
            pt50.append(preds["p50"] / 4.0)
            pt90.append(preds["p90"] / 4.0)
            sp.append(row.get("planned_spend_30", 0) / 4.0)
        if not yt:
            continue
        yt_a, pt10_a, pt50_a, pt90_a, sp_a = map(
            np.array, (yt, pt10, pt50, pt90, sp)
        )
        m = evaluate_predictions(
            yt_a,
            pt10_a,
            pt50_a,
            pt90_a,
            spend=sp_a,
            revenue_min=revenue_min,
            spend_min=spend_min,
            roas_cap=ROAS_CAP,
        )
        m["model_key"] = key
        by_type_records.append(m)
        y_true.extend(yt)
        p10.extend(pt10)
        p50.extend(pt50)
        p90.extend(pt90)
        spend.extend(sp)

    if not y_true:
        return {}

    y_true_a = np.array(y_true)
    p10_a = np.array(p10)
    p50_a = np.array(p50)
    p90_a = np.array(p90)
    spend_a = np.array(spend)

    overall = evaluate_predictions(
        y_true_a,
        p10_a,
        p50_a,
        p90_a,
        spend=spend_a,
        revenue_min=revenue_min,
        spend_min=spend_min,
        roas_cap=ROAS_CAP,
    )

    # Promote filtered metrics to primary (overwrite raw with filtered)
    for metric_name in [
        "wape_p50",      # PRIMARY (Fix 4)
        "smape_p50",
        "mape_p50",
        "mae_p50",
        "rmse_p50",
        "median_ae_p50",
        "coverage_p10_p90",
        "avg_interval_width",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "mape_roas",
        "mae_roas",
        "rmse_roas",
        "median_ae_roas",
    ]:
        filtered_key = f"filtered_{metric_name}"
        if filtered_key in overall:
            overall[f"raw_{metric_name}"] = overall[metric_name]
            overall[metric_name] = overall[filtered_key]

    overall["by_type"] = by_type_records
    overall["holdout_weeks"] = HOLDOUT_WEEKS
    overall["n_predictions"] = len(y_true_a)
    overall["revenue_filter_min"] = revenue_min
    overall["spend_filter_min"] = spend_min

    if by_type_records:
        diag_df = pd.DataFrame(by_type_records).copy()
        diag_df["mae_median_gap"] = diag_df["mae_p50"] - diag_df["median_ae_p50"]
        # Fix 4: Sort diagnostics by wape_p50 too (add fallback if column missing)
        # NOTE: mae_median_gap is the *primary* sort key, so low-spend groups
        # (e.g. bing|TOTAL, MAE~$447) rarely surface in the top-5 even when
        # their WAPE is poor.  Full per-group data is always in holdout_by_type.csv.
        sort_cols = [c for c in ["mae_median_gap", "wape_p50", "smape_p50"] if c in diag_df.columns]
        cols = [c for c in ["model_key", "wape_p50", "smape_p50", "mae_p50", "median_ae_p50",
                             "mae_median_gap", "coverage_p10_p90"] if c in diag_df.columns]
        overall["outlier_diagnostics"] = (
            diag_df.sort_values(sort_cols, ascending=False)[cols]
            .head(5)
            .to_dict(orient="records")
        )

    baseline = compare_to_baselines(
        test_df,
        y_true_a,
        p50_a,
        spend_a,
        revenue_min=revenue_min,
        spend_min=spend_min,
        roas_cap=ROAS_CAP,
    )
    overall["baseline_comparison"] = baseline["comparison"]
    overall["best_baseline"] = baseline["best_baseline"]
    # Fix 4: Report WAPE improvement as primary improvement metric
    overall["wape_improvement_vs_best_baseline"] = baseline["wape_improvement_vs_best_baseline"]
    overall["smape_improvement_vs_best_baseline"] = baseline["smape_improvement_vs_best_baseline"]
    overall["lightgbm_beats_baseline"] = baseline["lightgbm_beats_baseline"]
    overall["evaluation_notes"] = {
        "revenue_min": revenue_min,
        "revenue_min_quantile_raw": revenue_min_quantile,
        "revenue_min_floor": 500.0,
        "spend_min": spend_min,
        "roas_cap": ROAS_CAP,
        "filtered_rows": int(overall.get("filtered_n", len(y_true_a))),
        "total_rows": int(len(y_true_a)),
        "primary_metric": "WAPE",   # Fix 4: document that WAPE is primary
    }

    return overall


def ablation_daily_vs_weekly(data_dir: Path) -> pd.DataFrame:
    rows = []
    for freq in ["weekly", "daily"]:
        _, _, _, type_panel, _ = prepare_data(data_dir, freq=freq)
        type_panel = type_panel.copy()
        type_panel["model_key"] = type_panel.apply(lambda r: model_key(r), axis=1)
        bundle, test_df = train_quantile_models(type_panel)
        metrics = run_backtest(bundle, test_df)
        rows.append(
            {
                "aggregation": freq,
                "wape_p50": metrics.get("wape_p50", 0),   # PRIMARY (Fix 4)
                "smape_p50": metrics.get("smape_p50", 0),
                "mape_p50": metrics.get("mape_p50", 0),
                "mae_p50": metrics.get("mae_p50", 0),
                "rmse_p50": metrics.get("rmse_p50", 0),
                "coverage_p10_p90": metrics.get("coverage_p10_p90", 0),
                "avg_interval_width": metrics.get("avg_interval_width", 0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_at_campaign_level(bundle, panel_df, holdout_weeks):
    """
    Generate predictions at the individual campaign level for an apples-to-apples
    MAE comparison with campaign-level baselines.
    """
    test_start = panel_df["date"].max() - pd.Timedelta(weeks=holdout_weeks)
    test_panel = panel_df[panel_df["date"] >= test_start].copy()
    
    merged_rows = []
    for _, row in test_panel.iterrows():
        actual = float(row.get("target_30", 0)) / 4.0
        if actual < 100:
            continue
            
        preds = predict_group(bundle, row, horizon=30)
        p50 = preds["p50"] / 4.0
        
        merged_rows.append({
            "actual_revenue": actual,
            "p50_revenue": p50,
        })
        
    merged = pd.DataFrame(merged_rows)
    if merged.empty:
        return {}
        
    campaign_mae  = np.mean(np.abs(merged['actual_revenue'] - merged['p50_revenue']))
    campaign_wape = (np.sum(np.abs(merged['actual_revenue'] - merged['p50_revenue'])) /
                     np.sum(merged['actual_revenue'])) * 100

    return {
        "campaign_level_MAE":  float(campaign_mae),
        "campaign_level_WAPE": float(campaign_wape),
        "n_campaign_rows":     len(merged),
    }

def main():
    parser = argparse.ArgumentParser(description="Train AIgnition forecasting model v3.1")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--model-path", default=str(ROOT / "pickle" / "model.pkl"))
    parser.add_argument("--ablation", action="store_true", help="Run daily vs weekly ablation")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter tuning (Fix 12)")
    parser.add_argument("--cv", action="store_true", help="Run 3-fold walk-forward CV (Fix 13)")
    args = parser.parse_args()

    # Lock all random state for fully reproducible runs
    np.random.seed(42)
    import random as _random
    _random.seed(42)

    data_dir = Path(args.data_dir)
    _, report, panel, type_panel, channel_panel = prepare_data(data_dir, freq="weekly")

    # Enable native channel forecasting
    channel_panel = channel_panel.copy()
    channel_panel["campaign_type"] = "TOTAL"

    combined_panel = pd.concat([type_panel, channel_panel], ignore_index=True)
    combined_panel["model_key"] = combined_panel.apply(lambda r: model_key(r), axis=1)

    bundle, test_df = train_quantile_models(
        combined_panel,
        run_optuna=args.optuna,
        run_cv=args.cv,
    )
    bundle.validation_report = report.to_dict()
    bundle.meta_revenue_assumption = (
        "conversion_as_revenue" if META_CONVERSION_AS_REVENUE else "meta_excluded"
    )
    bundle.holdout_metrics = run_backtest(bundle, test_df)
    
    # Step 5: Campaign-level MAE
    print("[Eval] Running campaign-level evaluation pass...")
    camp_metrics = evaluate_at_campaign_level(bundle, panel, HOLDOUT_WEEKS)
    bundle.holdout_metrics.update(camp_metrics)

    # Print real feature importances
    try:
        # Explicitly grab Google TOTAL to represent macro portfolio drivers
        target_key = "google|TOTAL" if "google|TOTAL" in bundle.quantile_models else list(bundle.quantile_models.keys())[0]
        q50_model = bundle.quantile_models[target_key][30][0.5]
        
        importances = q50_model.feature_importance(importance_type='split')
        print(f"\n[Actual Feature Importances: {target_key}]")
        imp_df = pd.DataFrame({'Feature': bundle.feature_cols, 'Score': importances}).sort_values(by='Score', ascending=False)
        print(imp_df.head(15).to_string(index=False))
    except Exception as e:
        pass

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    if bundle.holdout_metrics:
        pd.DataFrame(bundle.holdout_metrics.get("by_type", [])).to_csv(
            docs / "holdout_by_type.csv", index=False
        )
        pd.DataFrame(bundle.holdout_metrics.get("baseline_comparison", [])).to_csv(
            docs / "baseline_comparison.csv", index=False
        )

        # Save CV results if available (Fix 13)
        cv_records = bundle.holdout_metrics.get("cv_results", [])
        if cv_records:
            pd.DataFrame(cv_records).to_csv(docs / "cv_results.csv", index=False)

        with open(docs / "validation_results.json", "w") as f:
            summary = {k: v for k, v in bundle.holdout_metrics.items()
                       if k not in ("by_type", "cv_results")}
            json.dump(summary, f, indent=2, default=str)

    if args.ablation:
        ablation = ablation_daily_vs_weekly(data_dir)
        ablation.to_csv(docs / "ablation_results.csv", index=False)
        print(f"Ablation saved to {docs / 'ablation_results.csv'}")

    save_bundle(bundle, args.model_path)
    hm = bundle.holdout_metrics

    # Results printout - WAPE is primary (Fix 4)
    print()
    print("=" * 60)
    print("  AIgnition v3.1 - Holdout Backtest Results")
    print("=" * 60)
    print(f"  Model:           {args.model_path}")
    print(f"  Holdout weeks:   {HOLDOUT_WEEKS} | predictions: {hm.get('n_predictions', 0)}")
    print(f"  Filtered rows:   {hm.get('filtered_n', 'N/A')} / {hm.get('n_predictions', 'N/A')}")
    print(f"  Revenue filter:  >= ${hm.get('revenue_filter_min', 0):,.2f}")
    print(f"  Spend filter:    >= ${hm.get('spend_filter_min', 0):.2f}")
    print(f"  ROAS cap:        {ROAS_CAP}x (Fix 14)")
    print()
    print("  -- PRIMARY METRIC --")
    print(f"  WAPE  (P50):     {hm.get('wape_p50', 'N/A'):.2f}%  <- PRIMARY (Fix 4)")
    print()
    print("  -- SECONDARY METRICS --")
    print(f"  SMAPE (P50):     {hm.get('smape_p50', 'N/A'):.2f}%")
    print(f"  SMAPE (Active):  {hm.get('smape_active_p50', 'N/A'):.2f}%  (P50 > $100)")
    print(f"  MAPE  (P50):     {hm.get('mape_p50', 'N/A'):.2f}%")
    print(f"  MAE   (P50):     ${hm.get('mae_p50', 'N/A'):,.2f}  (Group level)")
    print(f"  Median AE (P50): ${hm.get('median_ae_p50', 'N/A'):,.2f}")
    print(f"  RMSE  (P50):     ${hm.get('rmse_p50', 'N/A'):,.2f}")
    
    print()
    print("  -- CAMPAIGN LEVEL (Apples-to-Apples) --")
    if "campaign_level_MAE" in hm:
        print(f"  Campaign MAE:    ${hm['campaign_level_MAE']:,.2f}")
        print(f"  Campaign WAPE:   {hm['campaign_level_WAPE']:.2f}%")
        print(f"  Campaign rows:   {hm['n_campaign_rows']}")
    else:
        print("  Not evaluated")
    print()
    print("  -- INTERVAL QUALITY --")
    print(f"  Coverage P10-P90:{hm.get('coverage_p10_p90', 'N/A'):.1f}%  (target: >=80%)")
    print(f"  Avg Interval Wid:{hm.get('avg_interval_width', 'N/A'):,.0f}")
    print()
    print("  -- BASELINE COMPARISON --")
    if hm.get("outlier_diagnostics"):
        print(f"  Top outlier:     {hm['outlier_diagnostics'][0].get('model_key', 'N/A')}")
    print(f"  Best baseline:   {hm.get('best_baseline')}")
    print(f"  WAPE improve:    {hm.get('wape_improvement_vs_best_baseline', 0):+.2f} pp vs baseline")
    print(f"  SMAPE improve:   {hm.get('smape_improvement_vs_best_baseline', 0):+.2f} pp vs baseline")
    print(f"  Beats baseline:  {hm.get('lightgbm_beats_baseline')}")
    print("=" * 60)

    # Optuna params if used (Fix 12)
    if bundle.optuna_best_params:
        print(f"\n  [Optuna] Tuned params saved to bundle.")
        print(f"  Best params: {bundle.optuna_best_params}")

    # CV results summary if used (Fix 13)
    cv_records = hm.get("cv_results", [])
    if cv_records:
        cv_df = pd.DataFrame(cv_records)
        print(f"\n  -- WALK-FORWARD CV (Fix 13) --")
        for _, row in cv_df.iterrows():
            print(f"  Fold {int(row['fold'])}: WAPE={row['wape']:.2f}% | "
                  f"SMAPE={row['smape']:.2f}% | Coverage={row['coverage']:.1f}%")
        print(f"  Mean CV WAPE: {cv_df['wape'].mean():.2f}%")


if __name__ == "__main__":
    main()
