"""AI-assisted insights via Groq API.

Revisions vs v1:
- 4 distinct LLM insight functions: forecast_explanation, anomaly_interpretation, budget_recommendation, portfolio_insight
- Structured JSON output schemas enforced
- Graceful fallback for missing API key
- SHAP-like local importance computation for driver input
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import GROQ_API_URL, GROQ_MODEL
from src.model import predict_group

load_dotenv(override=True)


def _call_groq(prompt: str, api_key: str) -> dict | None:
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as exc:
        print(f"Groq API Error: {exc}")
    return None


def _compute_local_importance(bundle, row: pd.Series, campaign_type: str) -> list[tuple[str, float]]:
    """Pseudo-SHAP: Feature importance via one-at-a-time perturbation around P50."""
    from src.model import model_key, _encode_categories
    key = model_key(row, campaign_type)
    
    quantile_models = getattr(bundle, "quantile_models", {})
    if key not in quantile_models or 30 not in quantile_models[key]:
        return []
        
    category_maps = getattr(bundle, "category_maps", {})
    feature_cols = getattr(bundle, "feature_cols", [])
    if not feature_cols:
        return []
        
    df = pd.DataFrame([row])
    df, _ = _encode_categories(df, category_maps)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    X_base = df[feature_cols].copy()
    model = quantile_models[key][30][0.50]
    base_pred = float(model.predict(X_base)[0])
    
    importance = []
    # Perturb each feature by +10% or +1 unit
    for feat in feature_cols:
        if feat in ["channel", "campaign_type", "audience_segment"]:
            continue
        val = X_base.iloc[0][feat]
        delta = max(val * 0.1, 1.0)
        X_pert = X_base.copy()
        X_pert.at[0, feat] = val + delta
        new_pred = float(model.predict(X_pert)[0])
        impact = abs(new_pred - base_pred)
        importance.append((feat, impact))
        
    importance.sort(key=lambda x: x[1], reverse=True)
    return importance[:3]


def generate_forecast_explanation(
    forecasts: pd.DataFrame, bundle, type_panel: pd.DataFrame, api_key: str | None = None
) -> dict:
    """Function 1: Forecast Explanation with top-driver features."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    blended = forecasts[forecasts["level"] == "blended"].to_dict(orient="records")
    
    # Compute drivers for the top 3 campaign types
    top_types = forecasts[(forecasts["level"] == "campaign_type") & (forecasts["horizon_days"] == 30)].nlargest(3, "revenue_p50")
    drivers = []
    for _, row in top_types.iterrows():
        ct = row["campaign_type"]
        ch = row["channel"]
        recent_row = type_panel[(type_panel["channel"] == ch) & (type_panel["campaign_type"] == ct)].iloc[-1]
        imp = _compute_local_importance(bundle, recent_row, ct)
        
        # Calculate recent momentum for the LLM context
        ema_fast = recent_row.get("roll_mean_4", 1)
        ema_slow = recent_row.get("roll_mean_8", 1)
        momentum = "Accelerating" if ema_fast > ema_slow else "Decelerating"
        
        drivers.append({"group": f"{ch}|{ct}", "momentum": momentum, "top_features": [f[0] for f in imp]})
        
    if not api_key:
        return {"explanation": "Fallback: API key missing.", "drivers": drivers}

    prompt = f"""You are an elite, highly critical Chief Marketing Officer. Your job is NOT to summarize the numbers—the user can already see the charts. Your job is to extract the strategic narrative. 

Forecast data: {json.dumps(blended, indent=2)}
Top local features driving predictions: {json.dumps(drivers, indent=2)}

Analyze this data and output a JSON with this exact structure:
{{
  "explanation": "A punchy, 2-sentence executive summary of the portfolio's momentum and the hidden mathematical drivers behind it.",
  "biggest_risk": "Identify the single biggest vulnerability or diminishing return in the current strategy.",
  "biggest_opportunity": "Identify the single biggest untapped opportunity based on the feature importances (e.g., Cross-channel halo effects)."
}}
Do NOT use generic phrases like 'steady growth'. Be specific, contrarian, and actionable."""

    res = _call_groq(prompt, api_key)
    if res:
        res["drivers"] = drivers
        return res
    return {"explanation": "Fallback: API error.", "drivers": drivers}


def generate_anomaly_interpretation(
    anomalies: list[dict], api_key: str | None = None
) -> dict:
    """Function 2: Anomaly Interpretation."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not anomalies:
        return {"interpretation": "No significant anomalies found in the recent 52 weeks.", "actionable_advice": "Maintain course."}
        
    recent = anomalies[:5]
    if not api_key:
        return {"interpretation": f"Fallback: Found {len(anomalies)} anomalies.", "anomalies": recent}

    prompt = f"""You are an AI analyst. Review these historical anomalies where actual revenue fell outside the model's 80% confidence interval.
Anomalies: {json.dumps(recent, indent=2)}

Output JSON: {{"interpretation": "2 sentence summary of patterns in the anomalies", "actionable_advice": "1 sentence on how to handle these events in future planning"}}"""

    res = _call_groq(prompt, api_key)
    return res or {"interpretation": "Fallback: API error."}


def generate_budget_recommendation(
    curve_params: dict, target_roas: float = 3.0, api_key: str | None = None
) -> dict:
    """Function 3: Budget Recommendation based on curve shape."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    
    # In v3, all curves are Hill curves and params are stored as a list of 3 floats [vmax, k, n]
    saturating = {k: {"form": "hill", "params": v} for k, v in curve_params.items() if isinstance(v, list) and len(v) >= 3}
    
    if not api_key:
        return {"recommendation": "Fallback: API missing. Test +10% on saturating groups."}

    prompt = f"""You are a ruthless media buyer optimizing for a Target ROAS of {target_roas}. 
Review these spend-response curves (form and Hill parameters [vmax, k, n]): {json.dumps(saturating, indent=2)}

Identify exactly where budget is being wasted on saturating curves and where it should be reallocated.
Output JSON: 
{{
  "recommendation": "A 2-sentence aggressive reallocation strategy.",
  "recommended_shifts": [{{"group": "channel|type", "action": "Cut Spend | Hold | Scale Aggressively", "reason": "1-sentence mathematical justification"}}]
}}"""

    res = _call_groq(prompt, api_key)
    return res or {"recommendation": "Fallback: API error."}


def generate_portfolio_insight(
    forecasts: pd.DataFrame, api_key: str | None = None
) -> dict:
    """Function 4: Cross-Channel Portfolio Insight."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    
    ch_30 = forecasts[(forecasts["level"] == "channel") & (forecasts["horizon_days"] == 30)].to_dict(orient="records")
    
    if not api_key:
        return {"insight": "Fallback: API missing."}

    prompt = f"""You are a CMO. Provide a strategic portfolio insight comparing Google, Meta, and Bing based on this 30-day forecast.
Channel Forecasts: {json.dumps(ch_30, indent=2)}

Output JSON: {{"insight": "2 sentence strategic view on channel mix and efficiency (ROAS)", "strategic_focus": "The primary channel to focus on"}}"""

    res = _call_groq(prompt, api_key)
    return res or {"insight": "Fallback: API error."}


def generate_operational_risk(
    forecasts: pd.DataFrame, curve_params: dict, api_key: str | None = None
) -> dict:
    """Function 5: Operational Risk Assessment (Explicit hackathon requirement)."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    
    # Check for scenarios where simulated ROAS is < 1.0 or P10 revenue is dropping
    risk_candidates = forecasts[(forecasts["level"] == "campaign_type") & (forecasts["horizon_days"] == 30)].to_dict(orient="records")
    
    if not api_key:
        return {"risk_assessment": "Fallback: API missing."}

    prompt = f"""You are a risk management AI for a media buying agency. 
Review these 30-day campaign-level forecasts: {json.dumps(risk_candidates, indent=2)}

Identify any severe operational risks. Specifically look for:
1. Campaigns where P10 (pessimistic) revenue drops significantly.
2. Campaigns where ROAS is projected to drop below 1.0 (burning money).
Output JSON: {{"risk_assessment": "A 2-sentence urgent warning detailing the biggest financial risk to the portfolio based on the data", "action_required": "1 sentence immediate fix"}}"""

    res = _call_groq(prompt, api_key)
    return res or {"risk_assessment": "Fallback: API error."}


def chat_with_forecast(
    query: str, forecasts: pd.DataFrame, api_key: str | None = None
) -> str:
    """Agentic chat interface allowing interactive questions about the current state."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "I'm offline right now (GROQ_API_KEY is missing), but I'm ready to chat once configured!"
        
    blended = forecasts[forecasts["level"] == "blended"].to_dict(orient="records")
    channel = forecasts[forecasts["level"] == "channel"].to_dict(orient="records")
    
    # We don't use _call_groq here because we want raw text, not JSON
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are AIgnition, an expert AI CMO. Answer the user's question concisely based on the following forecast data. Be analytical and specific. Data: " + json.dumps({"blended": blended, "channel": channel})},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.5,
                "max_tokens": 512,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"Sorry, I encountered an error communicating with the API: {exc}"


def detect_anomalies(panel: pd.DataFrame, bundle) -> list[dict]:
    """Flag weeks where actual revenue fell outside model P10-P90 band."""
    anomalies = []
    for key in panel["model_key"].unique() if "model_key" in panel.columns else []:
        ct = key.split("|", 1)[1]
        subset = panel[panel["model_key"] == key].tail(52)
        for _, row in subset.iterrows():
            preds = predict_group(bundle, row, horizon=30, campaign_type=ct)
            actual = row["revenue"]
            p10_week = preds["p10"] / 4.0
            p90_week = preds["p90"] / 4.0
            if actual < p10_week or actual > p90_week:
                anomalies.append(
                    {
                        "date": str(row["date"].date()),
                        "channel": row["channel"],
                        "campaign_type": ct,
                        "actual_revenue": float(actual),
                        "p10": p10_week,
                        "p90": p90_week,
                        "direction": "below" if actual < p10_week else "above",
                    }
                )
    return sorted(anomalies, key=lambda x: x["date"], reverse=True)


def generate_all_insights(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
    type_panel: pd.DataFrame,
    bundle,
    api_key: str | None = None,
) -> dict:
    """Convenience function to run all 4 insights."""
    anomalies = detect_anomalies(type_panel, bundle)
    return {
        "forecast_explanation": generate_forecast_explanation(forecasts, bundle, type_panel, api_key),
        "anomaly_interpretation": generate_anomaly_interpretation(anomalies, api_key),
        "budget_recommendation": generate_budget_recommendation(bundle.curve_params, 3.0, api_key),
        "portfolio_insight": generate_portfolio_insight(forecasts, api_key),
        "operational_risk": generate_operational_risk(forecasts, bundle.curve_params, api_key),
        "raw_anomalies_sample": anomalies[:10]
    }