"""FastAPI backend for forecasting utility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import ROOT
from src.insights import detect_anomalies, generate_all_insights
from src.model import load_bundle
from src.pipeline import generate_forecasts, prepare_data

app = FastAPI(title="AIgnition Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = ROOT / "pickle" / "model.pkl"
_bundle = None
_cache: dict = {}


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained. Run: python src/train.py")
        _bundle = load_bundle(MODEL_PATH)
    return _bundle


class ForecastRequest(BaseModel):
    data_dir: str = "./data"
    horizons: list[int] = Field(default=[30, 60, 90])
    spend_scenario: float | None = None


class SimulateRequest(BaseModel):
    data_dir: str = "./data"
    horizon_days: int = 90
    total_budget: float


@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL_PATH.exists()}


@app.post("/validate")
def validate_data(data_dir: str = "./data"):
    _, report, _, _, _ = prepare_data(data_dir)
    return report.to_dict()


@app.post("/forecast")
def forecast(req: ForecastRequest):
    bundle = get_bundle()
    _, report, panel, type_panel, channel_panel = prepare_data(req.data_dir)
    forecasts = generate_forecasts(
        bundle,
        type_panel,
        channel_panel,
        panel,
        horizons=req.horizons,
        spend_scenario=req.spend_scenario,
    )
    return {"forecasts": forecasts.to_dict(orient="records"), "validation": report.to_dict()}


@app.post("/simulate")
def simulate(req: SimulateRequest):
    bundle = get_bundle()
    _, _, panel, type_panel, channel_panel = prepare_data(req.data_dir)
    recent = type_panel.sort_values("date").groupby(["channel", "campaign_type"]).tail(1)
    total_recent = recent["spend"].sum()
    scale = req.total_budget / total_recent if total_recent > 0 else 1.0
    avg_spend = recent["spend"].mean() * scale
    forecasts = generate_forecasts(
        bundle,
        type_panel,
        channel_panel,
        panel,
        horizons=[req.horizon_days],
        spend_scenario=avg_spend,
    )
    blended = forecasts[forecasts["level"] == "blended"]
    return {"forecasts": forecasts.to_dict(orient="records"), "budget_scale": scale, "blended": blended.to_dict(orient="records")}


@app.post("/insights")
def insights(data_dir: str = "./data"):
    bundle = get_bundle()
    _, report, panel, type_panel, channel_panel = prepare_data(data_dir)
    forecasts = generate_forecasts(bundle, type_panel, channel_panel, panel)
    from src.insights import generate_all_insights
    
    result = generate_all_insights(
        forecasts=forecasts,
        panel=panel,
        type_panel=type_panel,
        bundle=bundle
    )
    return result


@app.get("/metrics")
def metrics():
    bundle = get_bundle()
    return {"holdout_metrics": bundle.holdout_metrics, "curve_params": bundle.curve_params}
