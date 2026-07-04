"""Prediction entry point for run.sh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import ROOT
from src.model import load_bundle
from src.pipeline import generate_forecasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model", default=str(ROOT / "pickle" / "model.pkl"))
    parser.add_argument("--output", default=str(ROOT / "output" / "predictions.csv"))
    args = parser.parse_args()

    type_panel = pd.read_parquet(args.features)
    channel_path = Path(args.features).with_name("channel_features.parquet")
    campaign_path = Path(args.features).with_name("campaign_features.parquet")
    channel_panel = pd.read_parquet(channel_path)
    campaign_panel = pd.read_parquet(campaign_path)

    bundle = load_bundle(args.model)
    forecasts = generate_forecasts(
        bundle, type_panel, channel_panel, campaign_panel
    )

    # Reformat to match the required submission format exactly
    # Expected: channel,campaign_type,campaign_name,horizon_days,p10_revenue,p50_revenue,p90_revenue,p10_roas,p50_roas,p90_roas
    
    formatted = forecasts.rename(columns={
        "revenue_p10": "p10_revenue",
        "revenue_p50": "p50_revenue",
        "revenue_p90": "p90_revenue",
        "roas_p10": "p10_roas",
        "roas_p50": "p50_roas",
        "roas_p90": "p90_roas",
    })
    
    columns_to_keep = [
        "channel",
        "campaign_type",
        "campaign_name",
        "horizon_days",
        "p10_revenue",
        "p50_revenue",
        "p90_revenue",
        "p10_roas",
        "p50_roas",
        "p90_roas"
    ]
    
    formatted = formatted[columns_to_keep]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(out, index=False)
    print(f"Predictions written to {out} ({len(formatted)} rows)")


if __name__ == "__main__":
    main()
