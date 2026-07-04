"""Feature generation entry point for run.sh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import ROOT
from src.pipeline import prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out", default="features.parquet")
    args = parser.parse_args()

    _, report, panel, type_panel, channel_panel = prepare_data(args.data_dir, freq="weekly")

    features = {
        "campaign_type_panel": type_panel,
        "channel_panel": channel_panel,
        "campaign_panel": panel,
        "validation_report": pd.DataFrame([report.to_dict()]),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Store as parquet with campaign_type panel as primary
    type_panel.to_parquet(out_path, index=False)
    channel_panel.to_parquet(out_path.with_name("channel_features.parquet"), index=False)
    panel.to_parquet(out_path.with_name("campaign_features.parquet"), index=False)
    pd.DataFrame([report.to_dict()]).to_parquet(
        out_path.with_name("validation_report.parquet"), index=False
    )
    print(f"Features written to {out_path}")


if __name__ == "__main__":
    main()
