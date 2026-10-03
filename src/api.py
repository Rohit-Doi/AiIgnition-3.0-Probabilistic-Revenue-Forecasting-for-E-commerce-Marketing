"""FastAPI backend: the same engine the dashboard uses, over HTTP.

    uvicorn src.api:app --port 8000
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import scenario
from src.config import HORIZONS
from src.forecast import public_columns
from src.insights import available_providers
from src.service import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, Workspace, analyse, load_workspace, parse_multipliers

app = FastAPI(title="AIgnition Forecast API", version="4.0.0",
              description="Probabilistic revenue and ROAS forecasting for Google, Microsoft and Meta Ads.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=4)
def _workspace(data_dir: str) -> Workspace:
    return load_workspace(data_dir, DEFAULT_MODEL_PATH)


def workspace(data_dir: str) -> Workspace:
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run: python -m src.train")
    try:
        return _workspace(data_dir)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def _records(df: pd.DataFrame) -> list[dict]:
    return public_columns(df).assign(origin=lambda d: d["origin"].astype(str)).to_dict(orient="records")


class DataRequest(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)


class ForecastRequest(DataRequest):
    horizons: list[int] = Field(default=list(HORIZONS))
    levels: list[str] = Field(default=["blended", "channel", "campaign_type", "campaign"])


class SimulateRequest(DataRequest):
    horizon_days: int = 30
    multipliers: dict[str, float] = Field(
        default_factory=dict,
        description="Budget multiplier vs expected spend, keyed 'channel' or 'channel|CAMPAIGN_TYPE'.",
        examples=[{"google|SEARCH": 1.2, "meta": 0.8}])
    default_multiplier: float = 1.0


class OptimiseRequest(DataRequest):
    horizon_days: int = 30
    total_budget: float | None = Field(default=None, description="Defaults to the expected spend.")
    min_multiplier: float = 0.5
    max_multiplier: float = 2.0


class InsightRequest(DataRequest):
    horizon_days: int = 30
    use_llm: bool = True


def _check_horizon(h: int) -> None:
    if h not in HORIZONS:
        raise HTTPException(422, f"horizon_days must be one of {HORIZONS}")


@app.get("/health")
def health():
    return {"status": "ok", "model_exists": DEFAULT_MODEL_PATH.exists(), "llm_providers": available_providers()}


@app.post("/validate")
def validate_data(req: DataRequest):
    return workspace(req.data_dir).report


@app.post("/forecast")
def forecast(req: ForecastRequest):
    ws = workspace(req.data_dir)
    fc = ws.baseline
    fc = fc[fc["horizon_days"].isin(req.horizons) & fc["level"].isin(req.levels)]
    return {"origin": str(ws.hier.origin.date()), "forecasts": _records(fc), "validation": ws.report}


@app.post("/simulate")
def simulate(req: SimulateRequest):
    _check_horizon(req.horizon_days)
    ws = workspace(req.data_dir)
    scen = scenario.run_scenario(ws.baseline, parse_multipliers(req.multipliers), req.default_multiplier)
    pick = lambda d: d[(d["horizon_days"] == req.horizon_days) & (d["level"] != "campaign")]
    return {
        "baseline": _records(pick(ws.baseline)),
        "scenario": _records(pick(scen)),
        "extrapolation_warnings": scenario.extrapolation_warnings(ws.hier, scen, req.horizon_days),
    }


@app.post("/optimise")
def optimise(req: OptimiseRequest):
    _check_horizon(req.horizon_days)
    ws = workspace(req.data_dir)
    alloc = scenario.optimise_budget(ws.baseline, req.horizon_days, req.total_budget,
                                     bounds=(req.min_multiplier, req.max_multiplier))
    return {
        "allocation": alloc.to_dict(orient="records"),
        "revenue_current": float(alloc["revenue_current"].sum()) if len(alloc) else 0.0,
        "revenue_recommended": float(alloc["revenue_recommended"].sum()) if len(alloc) else 0.0,
    }


@app.post("/insights")
def insights(req: InsightRequest):
    _check_horizon(req.horizon_days)
    ws = workspace(req.data_dir)
    result = analyse(ws, req.horizon_days, use_llm=req.use_llm)
    return {
        "insights": result["insights"],
        "risks": result["risks"],
        "anomalies": result["anomalies"].head(20).to_dict(orient="records"),
        "facts": result["facts"],
    }


@app.get("/metrics")
def metrics():
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run: python -m src.train")
    from src.model import load_bundle

    bundle = load_bundle(DEFAULT_MODEL_PATH)
    backtest = {k: v for k, v in bundle.get("backtest", {}).items() if k != "replay"}
    return {"version": bundle["version"], "trained_on": bundle["trained_on"], "backtest": backtest,
            "elasticity": bundle["elasticity"]}
