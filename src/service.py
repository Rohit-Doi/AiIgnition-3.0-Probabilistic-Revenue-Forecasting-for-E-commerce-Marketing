"""One place where the dashboard and the API assemble a full analysis.

A Workspace is a data folder that has been ingested, validated and forecast with
a model bundle.  Everything interactive (scenarios, optimiser, insights) is
derived from it without touching the model again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import explain, scenario
from src.config import HORIZONS, ROOT
from src.forecast import forecast
from src.insights import generate_insights
from src.model import load_bundle
from src.pipeline import Prepared, feature_table, prepare_data

DEFAULT_MODEL_PATH = ROOT / "pickle" / "model.pkl"
DEFAULT_DATA_DIR = ROOT / "data"


@dataclass
class Workspace:
    prepared: Prepared
    bundle: dict
    baseline: pd.DataFrame          # forecast table incl. internal helper columns

    @property
    def hier(self):
        return self.prepared.hier

    @property
    def report(self) -> dict:
        return self.prepared.report.to_dict()


def load_workspace(data_dir: Path | str = DEFAULT_DATA_DIR,
                   model_path: Path | str = DEFAULT_MODEL_PATH) -> Workspace:
    bundle = load_bundle(model_path)
    prepared = prepare_data(data_dir)
    return Workspace(prepared=prepared, bundle=bundle,
                     baseline=forecast(bundle, feature_table(prepared.hier)))


def parse_multipliers(raw: dict[str, float] | None) -> dict[tuple[str, str], float]:
    """{'google|SEARCH': 1.2, 'meta': 0.8} -> {('google','SEARCH'): 1.2, ('meta','*'): 0.8}."""
    out: dict[tuple[str, str], float] = {}
    for key, value in (raw or {}).items():
        channel, _, ctype = key.partition("|")
        out[(channel.strip().lower(), ctype.strip().upper() or "*")] = float(value)
    return out


def analyse(ws: Workspace, horizon: int = HORIZONS[0], fc: pd.DataFrame | None = None,
            with_optimiser: bool = True, use_llm: bool = True,
            optimiser_bounds: tuple[float, float] = (0.5, 2.0)) -> dict:
    """Anomalies, risks, optimiser result, fact sheet and written insights for one horizon.

    `fc` is the forecast being discussed: the baseline, or a scenario built on it.
    """
    is_scenario = fc is not None and (fc["mode"] == "scenario").any()
    current = fc if fc is not None else ws.baseline
    anomalies = explain.detect_anomalies(ws.hier)
    risks = explain.risk_flags(current, ws.hier, ws.report, horizon)
    allocation = (scenario.optimise_budget(ws.baseline, horizon, bounds=optimiser_bounds)
                  if with_optimiser else None)
    facts = explain.build_facts(ws.baseline, ws.hier, ws.report, ws.bundle, horizon,
                                scenario=current if is_scenario else None,
                                allocation=allocation, anomalies=anomalies, risks=risks)
    return {
        "facts": facts,
        "anomalies": anomalies,
        "risks": risks,
        "allocation": allocation,
        "insights": generate_insights(facts, use_llm=use_llm),
    }
