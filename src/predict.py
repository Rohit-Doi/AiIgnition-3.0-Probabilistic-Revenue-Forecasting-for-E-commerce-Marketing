"""Prediction entry point for run.sh: features + pickled model -> predictions.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import ROOT
from src.forecast import forecast, public_columns, to_submission
from src.model import load_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model", default=str(ROOT / "pickle" / "model.pkl"))
    parser.add_argument("--output", default=str(ROOT / "output" / "predictions.csv"))
    args = parser.parse_args()

    feats = pd.read_parquet(args.features)
    bundle = load_bundle(args.model)
    fc = forecast(bundle, feats)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = to_submission(fc)
    if submission[["p10_revenue", "p50_revenue", "p90_revenue"]].isna().any().any():
        raise ValueError("Forecast contains missing values; refusing to write a partial file.")
    submission.to_csv(out, index=False)
    # Same forecasts with spend, drivers and level labels, for the dashboard and analysts.
    public_columns(fc).to_csv(out.with_name("forecast_detail.csv"), index=False)

    blended = fc[fc["level"] == "blended"]
    for _, r in blended.iterrows():
        print(f"  {int(r['horizon_days'])}d blended revenue  P10 {r['revenue_p10']:>12,.0f}  "
              f"P50 {r['revenue_p50']:>12,.0f}  P90 {r['revenue_p90']:>12,.0f}  |  ROAS P50 {r['roas_p50']:.2f}")
    print(f"Predictions written to {out} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
