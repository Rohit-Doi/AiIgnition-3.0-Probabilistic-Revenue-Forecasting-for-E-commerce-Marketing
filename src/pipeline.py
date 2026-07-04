"""End-to-end forecasting pipeline with Monte Carlo horizon aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregate import aggregate_by_level, build_training_panel
from src.config import HORIZONS, META_CONVERSION_AS_REVENUE, QUANTILE_LABELS, MC_SAMPLES
from src.load_data import load_all
from src.model import ModelBundle, load_bundle, predict_group
from src.reconcile import compute_roas, reconcile_quantiles
from src.validate import clean, validate


def prepare_data(data_dir: Path | str, freq: str = "weekly"):
    raw = load_all(data_dir)
    report = validate(raw)
    cleaned = clean(raw, report)
    panel = build_training_panel(cleaned, freq=freq)
    type_panel = aggregate_by_level(panel, "campaign_type")
    channel_panel = aggregate_by_level(panel, "channel")
    return cleaned, report, panel, type_panel, channel_panel


def _weeks_for_horizon(days: int) -> float:
    return days / 7.0


def _latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    keys = ["channel"]
    if "campaign_type" in panel.columns:
        keys.append("campaign_type")
    return (
        panel.sort_values("date")
        .groupby(keys, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _history_for_row(panel: pd.DataFrame, row: pd.Series, keys: list[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=panel.index)
    for key in keys:
        mask &= panel[key] == row[key]
    return panel.loc[mask].sort_values("date").tail(12).reset_index(drop=True)


def optimal_bottom_up_reconciliation(predictions_dict: dict[str, float]) -> dict[str, float]:
    """
    Blueprint Implementation: Hierarchical Reconciliation Layer
    predictions_dict layout: { 'SEARCH': 1200, 'SHOPPING': 800, 'TOTAL': 2400 }
    Forces sub-nodes to reconcile linearly to valid sum configurations.
    """
    google_sum = sum(v for k, v in predictions_dict.items() if k != "TOTAL")
    google_total = predictions_dict.get("TOTAL", google_sum)
    
    discrepancy = google_total - google_sum
    if google_sum > 0 and discrepancy != 0:
        for k in predictions_dict:
            if k != "TOTAL":
                predictions_dict[k] += discrepancy * (predictions_dict[k] / google_sum)
    return predictions_dict


def generate_forecasts(
    bundle: ModelBundle,
    type_panel: pd.DataFrame,
    channel_panel: pd.DataFrame,
    panel: pd.DataFrame,
    horizons: list[int] | None = None,
    spend_scenario: float | None = None,
) -> pd.DataFrame:
    horizons = horizons or HORIZONS
    latest = _latest_rows(type_panel)
    latest_channels = _latest_rows(channel_panel)
    rows = []

    type_preds: dict[str, dict[str, float]] = {}
    type_spends: dict[str, float] = {}

    for _, row in latest.iterrows():
        ct = row["campaign_type"]
        ch = row["channel"]
        base_spend = float(row["spend"])
        spend_per_week = spend_scenario if spend_scenario else base_spend

        for horizon in horizons:
            weeks = _weeks_for_horizon(horizon)
            total_spend = spend_per_week * weeks
            
            # Predict directly with the Direct Horizon model!
            scaled = predict_group(
                bundle, 
                row, 
                horizon=horizon, 
                campaign_type=ct, 
                spend_override=total_spend
            )
            
            key = f"{ch}|{ct}|{horizon}"
            type_preds[key] = scaled
            type_spends[key] = total_spend
            roas = compute_roas(scaled, total_spend)

            rows.append(
                {
                    "horizon_days": horizon,
                    "level": "campaign_type",
                    "channel": ch,
                    "campaign_type": ct,
                    "campaign_id": "",
                    "campaign_name": "",
                    "spend_scenario": total_spend,
                    "revenue_p10": scaled["p10"],
                    "revenue_p50": scaled["p50"],
                    "revenue_p90": scaled["p90"],
                    "roas_p10": roas["p10"],
                    "roas_p50": roas["p50"],
                    "roas_p90": roas["p90"],
                }
            )

    # Channel level reconciliation
    for horizon in horizons:
        for ch in type_panel["channel"].unique():
            # 1. Gather bottom-up predictions
            group_preds = {
                k.split("|")[1]: type_preds[k]
                for k in type_preds
                if k.endswith(f"|{horizon}") and k.startswith(f"{ch}|")
            }
            if not group_preds:
                continue

            # 2. Get native 'TOTAL' prediction for this channel
            ch_row = latest_channels[latest_channels["channel"] == ch]
            total_spend = sum(
                type_spends[k]
                for k in type_preds
                if k.endswith(f"|{horizon}") and k.startswith(f"{ch}|")
            )
            
            if len(ch_row) > 0 and (f"{ch}|TOTAL" in getattr(bundle, "quantile_models", {})):
                # The model was trained with campaign_type = "TOTAL"
                native_total = predict_group(
                    bundle, 
                    ch_row.iloc[0], 
                    horizon=horizon, 
                    campaign_type="TOTAL", 
                    spend_override=total_spend
                )
            else:
                # Fallback if no native total model exists
                native_total = reconcile_quantiles(list(group_preds.values()))

            # 3. Apply Hierarchical Reconciliation Layer
            for q in ["p10", "p50", "p90"]:
                pred_dict = {ct: preds[q] for ct, preds in group_preds.items()}
                pred_dict["TOTAL"] = native_total[q]
                
                reconciled_dict = optimal_bottom_up_reconciliation(pred_dict)
                
                # Write back reconciled values
                for ct in group_preds:
                    group_preds[ct][q] = reconciled_dict[ct]
                native_total[q] = reconciled_dict["TOTAL"]

            # Update the original rows with reconciled values
            for row in rows:
                if row["channel"] == ch and row["horizon_days"] == horizon and row["level"] == "campaign_type":
                    ct = row["campaign_type"]
                    row["revenue_p10"] = group_preds[ct]["p10"]
                    row["revenue_p50"] = group_preds[ct]["p50"]
                    row["revenue_p90"] = group_preds[ct]["p90"]

            roas = compute_roas(native_total, total_spend)
            rows.append(
                {
                    "horizon_days": horizon,
                    "level": "channel",
                    "channel": ch,
                    "campaign_type": "",
                    "campaign_id": "",
                    "campaign_name": "",
                    "spend_scenario": total_spend,
                    "revenue_p10": native_total["p10"],
                    "revenue_p50": native_total["p50"],
                    "revenue_p90": native_total["p90"],
                    "roas_p10": roas["p10"],
                    "roas_p50": roas["p50"],
                    "roas_p90": roas["p90"],
                }
            )

    # Blended total
    for horizon in horizons:
        ch_preds = [
            {
                "p10": r["revenue_p10"],
                "p50": r["revenue_p50"],
                "p90": r["revenue_p90"],
            }
            for r in rows
            if r["level"] == "channel" and r["horizon_days"] == horizon
        ]
        total_spend = sum(r["spend_scenario"] for r in rows if r["level"] == "channel" and r["horizon_days"] == horizon)
        if ch_preds:
            blended = reconcile_quantiles(ch_preds)
            roas = compute_roas(blended, total_spend)
            rows.append(
                {
                    "horizon_days": horizon,
                    "level": "blended",
                    "channel": "all",
                    "campaign_type": "",
                    "campaign_id": "",
                    "campaign_name": "",
                    "spend_scenario": total_spend,
                    "revenue_p10": blended["p10"],
                    "revenue_p50": blended["p50"],
                    "revenue_p90": blended["p90"],
                    "roas_p10": roas["p10"],
                    "roas_p50": roas["p50"],
                    "roas_p90": roas["p90"],
                }
            )

    # Campaign level (top campaigns by recent revenue)
    campaign_latest = panel.sort_values("date").groupby(["channel", "campaign_id"]).tail(1)
    top = campaign_latest.nlargest(20, "revenue")
    for _, row in top.iterrows():
        ct = row["campaign_type"]
        for horizon in horizons:
            spend = float(row["spend"])
            weeks = _weeks_for_horizon(horizon)
            total_spend = spend * weeks
            
            scaled = predict_group(
                bundle, 
                row, 
                horizon=horizon, 
                campaign_type=ct, 
                spend_override=total_spend
            )
            roas = compute_roas(scaled, total_spend)
            rows.append(
                {
                    "horizon_days": horizon,
                    "level": "campaign",
                    "channel": row["channel"],
                    "campaign_type": ct,
                    "campaign_id": row["campaign_id"],
                    "campaign_name": row["campaign_name"],
                    "spend_scenario": total_spend,
                    "revenue_p10": scaled["p10"],
                    "revenue_p50": scaled["p50"],
                    "revenue_p90": scaled["p90"],
                    "roas_p10": roas["p10"],
                    "roas_p50": roas["p50"],
                    "roas_p90": roas["p90"],
                }
            )

    return pd.DataFrame(rows)
