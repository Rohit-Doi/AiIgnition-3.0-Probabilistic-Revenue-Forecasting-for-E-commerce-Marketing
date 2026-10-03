"""Feature generation entry point for run.sh.

Reads whatever channel CSVs are in the data folder and writes one feature row
per (series, horizon) at the forecast origin - everything predict.py needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT
from src.pipeline import feature_table, prepare_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out", default="features.parquet")
    args = parser.parse_args()

    prepared = prepare_data(args.data_dir)
    feats = feature_table(prepared.hier)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    with open(out_path.with_name("validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(prepared.report.to_dict(), f, indent=2)

    hier = prepared.hier
    print(f"Data: {hier.dates[0].date()} to {hier.origin.date()} | "
          f"{hier.n_series} series ({(hier.meta['level'] == 'campaign').sum()} campaigns)")
    print(f"Features written to {out_path} ({len(feats)} rows)")


if __name__ == "__main__":
    main()
