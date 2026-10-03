"""AI-assisted interpretation of the forecast (dashboard / API only).

The LLM never sees raw rows and never produces a number of its own.  It receives
the fact sheet built by src/explain.py - forecasts, the model's own driver
decomposition, anomalies, risks, marginal ROAS, backtest reliability - and is
asked to reason about causes and actions.  Its answer is then checked: every
number it quotes must be traceable to the fact sheet.

Works with any OpenAI-compatible chat endpoint (Groq, Gemini, OpenAI).  Keys are
read from the environment / .env; without a working key a deterministic writer
produces the same sections from the same facts, so the product never shows an
empty panel.  run.sh never imports this module.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache

import requests

from src.config import LLM_PROVIDERS, LLM_TIMEOUT_SECONDS, ROOT

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:      # python-dotenv is optional
    pass

SYSTEM_PROMPT = """You are a senior performance-marketing analyst writing for an agency account team.
You are given a JSON fact sheet produced by a forecasting model for an ecommerce advertiser
(Google Ads, Microsoft Ads, Meta Ads). Rules:
1. Use ONLY the fact sheet. Never invent a number. Quote numbers exactly as they appear
   (you may round dollars to the nearest thousand and write them as $251k).
2. Separate what the model attributes from what you infer. `drivers_of_blended_forecast` is the
   model's exact decomposition of the forecast: cite it as "model-attributed". Real-world causes
   you propose (a promotion ending, budget pull-back, tracking loss, seasonality) are "hypothesis"
   and must be consistent with the anomalies, spend and revenue history in the fact sheet.
3. Be specific and operational: name the channel / campaign type, the number, and the action.
4. Revenue and ROAS are ranges. Refer to P10-P90 when discussing risk; never present the median
   as certain.
5. Write for a reader who never sees the fact sheet: plain language, never JSON field names
   (write "-$6,545, or -1.6%", not "effect_usd -6545").
6. No filler, no hedging boilerplate, no restating the task."""

INSIGHT_SCHEMA = """Return a JSON object with exactly these keys:
{
  "headline": "one sentence, max 22 words, the single most decision-relevant message",
  "executive_summary": "3-4 sentences: expected revenue and ROAS with range, direction vs run-rate, why",
  "causal_drivers": [
    {"driver": "short name", "direction": "up|down", "evidence": "the numbers from the fact sheet",
     "likely_cause": "business explanation", "basis": "model-attributed|hypothesis"}
  ],
  "anomalies": [
    {"what": "which week / channel / metric with numbers", "interpretation": "most plausible cause",
     "action": "what to check or do"}
  ],
  "risks": [
    {"risk": "specific risk", "severity": "high|medium|low", "mitigation": "concrete step"}
  ],
  "recommendations": [
    {"action": "specific budget or operational move", "rationale": "numbers that justify it",
     "expected_impact": "from the fact sheet if available"}
  ],
  "confidence": "1-2 sentences on how far to trust this forecast, using model_reliability"
}
Give 3-5 causal_drivers, 0-4 anomalies (only real ones from recent_anomalies), 2-4 risks,
2-4 recommendations. Use marginal_roas / budget_optimiser for budget recommendations."""


# ─── Provider plumbing ───────────────────────────────────────────────────────

def _candidate_keys(provider: str) -> list[str]:
    """Every configured key for a provider (primary first, then *_FALLBACK variants)."""
    keys = []
    for env in LLM_PROVIDERS[provider]["env"]:
        for name in (env, f"{env}_FALLBACK", f"{env}_2"):
            value = (os.getenv(name) or "").strip()
            if value and value not in keys:
                keys.append(value)
    return keys


def available_providers() -> list[str]:
    """Providers with a key configured. AIGNITION_DISABLE_LLM=1 forces the rule-based writer."""
    if os.getenv("AIGNITION_DISABLE_LLM", "").strip() in ("1", "true", "yes"):
        return []
    return [p for p in LLM_PROVIDERS if _candidate_keys(p)]


@lru_cache(maxsize=16)
def _usable_models(provider: str, key: str) -> tuple[str, ...]:
    """Preferred models this key can actually use, in order (hosted line-ups change).

    Empty when the key is rejected.  LLM_MODEL forces a single model.
    """
    cfg = LLM_PROVIDERS[provider]
    override = os.getenv("LLM_MODEL")
    if override:
        return (override,)
    try:
        r = requests.get(cfg["models_url"], headers={"Authorization": f"Bearer {key}"}, timeout=15)
        if r.status_code in (401, 403):
            return ()
        served = {m.get("id", "").removeprefix("models/") for m in r.json().get("data", [])} if r.ok else set()
    except (requests.RequestException, ValueError):
        served = set()
    usable = tuple(m for m in cfg["models"] if not served or m in served)
    return usable or (cfg["models"][0],)


def _post(provider: str, key: str, model: str, messages: list[dict], json_mode: bool,
          max_tokens: int, temperature: float) -> tuple[str | None, str, float]:
    """One chat call. Returns (text, error, seconds the provider asked us to wait)."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"      # keep hidden reasoning from eating the token budget
    attempts = [{**body, "response_format": {"type": "json_object"}}, body] if json_mode else [body]
    error = ""
    for payload in attempts:
        try:
            r = requests.post(LLM_PROVIDERS[provider]["url"], json=payload, timeout=LLM_TIMEOUT_SECONDS,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        except requests.RequestException as exc:
            return None, f"{provider}/{model}: {type(exc).__name__}", 0.0
        if r.ok:
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            return (text, "", 0.0) if text else (None, f"{provider}/{model}: empty response", 0.0)
        error = f"{provider}/{model}: HTTP {r.status_code}"
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("retry-after", 0) or 0)
            except ValueError:
                wait = 0.0
            return None, error + " (rate limited)", wait
        if r.status_code != 400:              # only a rejected parameter is worth a second attempt
            break
    return None, error, 0.0


def call_llm(messages: list[dict], json_mode: bool = False, max_tokens: int = 2000,
             temperature: float = 0.2, max_wait: float = 25.0) -> dict:
    """Try providers, keys and models in order; returns {text, provider, model} or {error}.

    A rate-limited model is skipped in favour of the next one; if everything is
    rate limited the shortest requested wait is honoured once (up to `max_wait`).
    """
    errors, waits = [], []
    targets = [(p, k, m) for p in available_providers() for k in _candidate_keys(p) for m in _usable_models(p, k)]
    if not targets:
        return {"error": "no working LLM key configured"}
    for round_no in range(2):
        for provider, key, model in targets:
            text, error, wait = _post(provider, key, model, messages, json_mode, max_tokens, temperature)
            if text:
                return {"text": text, "provider": provider, "model": model}
            errors.append(error)
            if wait:
                waits.append(wait)
        if round_no == 0 and waits and min(waits) <= max_wait:
            time.sleep(min(waits) + 0.5)
        else:
            break
    return {"error": "; ".join(dict.fromkeys(errors))}


def _parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ─── Grounding check ─────────────────────────────────────────────────────────

_NUMBER = re.compile(r"(?<![\w.])(-?\$?\d[\d,]*\.?\d*)\s*(k|m|bn|b|%|x)?(?![\w])", re.IGNORECASE)
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9}


def _fact_numbers(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _fact_numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in _fact_numbers(v)]
    if isinstance(obj, str):
        return [float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\.?\d*", obj) if m.strip(",")]
    return []


def check_grounding(texts: list[str], facts: dict) -> dict:
    """Which numbers quoted by the LLM can be traced to the fact sheet.

    Small counts (<= 12), calendar numbers and years are ignored.  A quoted number
    is accepted when a fact matches it after the same rounding the text used.
    """
    known = [abs(n) for n in _fact_numbers(facts)]
    total, unverified = 0, []
    for text in texts:
        for raw, suffix in _NUMBER.findall(text or ""):
            cleaned = raw.replace("$", "").replace(",", "")
            if not cleaned or cleaned in {"-", "."}:
                continue
            try:
                value = abs(float(cleaned))
            except ValueError:
                continue
            suffix = (suffix or "").lower()
            value *= _SCALE.get(suffix, 1.0)
            if value <= 12 and suffix not in ("%", "x") and "." not in cleaned:
                continue
            if suffix == "" and float(cleaned).is_integer() and (value in (14, 28, 30, 56, 60, 90, 91, 182)
                                                                  or 1990 <= value <= 2100):
                continue
            total += 1
            decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
            step = _SCALE.get(suffix, 1.0) * 10 ** (-decimals)
            tol = max(step * 0.51, value * 0.006)
            if not any(abs(value - k) <= tol for k in known):
                unverified.append(f"{raw}{suffix}")
    return {"numbers_checked": total, "unverified": sorted(set(unverified))}


# ─── Deterministic writer (no LLM) ───────────────────────────────────────────

def _usd(x: float) -> str:
    return f"${x:,.0f}"


def rule_based_insights(facts: dict) -> dict:
    """The same sections, written from the fact sheet by templates."""
    b = facts["blended"]
    h = facts["horizon_days"]
    direction = "above" if b["vs_run_rate_pct"] >= 0 else "below"
    drivers = sorted(facts.get("drivers_of_blended_forecast", []), key=lambda d: -abs(d["effect_usd"]))
    top = drivers[0] if drivers else None

    summary = (
        f"Over the next {h} days ({facts['window']}) blended revenue is expected at {_usd(b['revenue_p50'])}, "
        f"with an 80% range of {_usd(b['revenue_p10'])} to {_usd(b['revenue_p90'])}, on about "
        f"{_usd(b['spend_p50'])} of spend (ROAS {b['roas_p50']:.2f}, range {b['roas_p10']:.2f}-{b['roas_p90']:.2f}). "
        f"That is {abs(b['vs_run_rate_pct']):.1f}% {direction} the last-28-day run-rate of {_usd(b['run_rate_revenue'])}."
    )
    if top:
        summary += f" The largest single driver is {top['driver'].lower()} ({top['effect_pct']:+.1f}%, {_usd(top['effect_usd'])})."

    causal = [{
        "driver": d["driver"], "direction": "up" if d["effect_usd"] >= 0 else "down",
        "evidence": f"{d['effect_pct']:+.1f}% ({_usd(d['effect_usd'])}) on the blended {h}-day forecast",
        "likely_cause": "Taken directly from the model's decomposition of the forecast.",
        "basis": "model-attributed",
    } for d in drivers[:5]]

    anomalies = [{
        "what": f"Week ending {a['week_ending']}: {a['where']} {a['metric']} was {a['value']:,.2f} "
                f"vs a typical {a['typical']:,.2f} ({a['direction']})",
        "interpretation": ("Spend continued while attributed revenue fell: check tracking and attribution first."
                           if a["metric"] == "roas" and a["direction"] == "below"
                           else "A break from the prior eight weeks; confirm whether a promotion or budget change explains it."),
        "action": "Annotate the cause so future forecasts can be read against it.",
    } for a in facts.get("recent_anomalies", [])[:4]]

    risks = [{"risk": f"{r['title']}: {r['detail']}", "severity": r["severity"],
              "mitigation": "Plan budgets against the P10-P90 range and re-forecast weekly."}
             for r in facts.get("risks", [])[:4]]

    recs = []
    opt = facts.get("budget_optimiser")
    if opt and opt.get("moves"):
        up = [m for m in opt["moves"] if m["change_pct"] > 0][:2]
        down = [m for m in opt["moves"] if m["change_pct"] < 0][:2]
        if up and down:
            recs.append({
                "action": "Shift budget from " + ", ".join(f"{m['channel']} {m['campaign_type']}" for m in down)
                          + " to " + ", ".join(f"{m['channel']} {m['campaign_type']}" for m in up),
                "rationale": "Marginal ROAS is "
                             + ", ".join(f"{m['marginal_roas_before']:.2f} for {m['channel']} {m['campaign_type']}" for m in up + down),
                "expected_impact": f"{_usd(opt['expected_revenue_gain'])} ({opt['expected_revenue_gain_pct']:+.1f}%) "
                                   f"more revenue on the same {_usd(opt['total_budget'])} budget",
            })
    mr = facts.get("marginal_roas", [])
    if len(mr) >= 2 and not recs:
        best, worst = mr[0], mr[-1]
        recs.append({
            "action": f"Favour {best['channel']} {best['campaign_type']} over {worst['channel']} {worst['campaign_type']} at the margin",
            "rationale": f"Marginal ROAS {best['marginal_roas']:.2f} vs {worst['marginal_roas']:.2f}",
            "expected_impact": "Use the budget optimiser to size the move.",
        })
    recs.append({
        "action": "Lock a budget plan in the simulator before committing to a revenue number",
        "rationale": "A large share of forecast error comes from not knowing future spend.",
        "expected_impact": (f"Backtest error falls from {facts['model_reliability']['blended_wape_pct']:.1f}% to "
                            f"{facts['model_reliability']['blended_wape_if_budget_known_pct']:.1f}% when the budget is known"
                            if "model_reliability" in facts else "Narrower range"),
    })

    rel = facts.get("model_reliability")
    confidence = (
        f"On {rel['backtest_origins']} past forecasts of this horizon the typical miss was {rel['blended_wape_pct']:.1f}% of "
        f"actual revenue ({rel['blended_wape_if_budget_known_pct']:.1f}% when the budget was known) and the P10-P90 range "
        f"contained the outcome {rel['interval_coverage_pct']:.1f}% of the time."
        if rel else "No backtest is stored with this model.")

    return {
        "headline": f"{h}-day revenue expected at {_usd(b['revenue_p50'])}, {abs(b['vs_run_rate_pct']):.1f}% {direction} run-rate.",
        "executive_summary": summary, "causal_drivers": causal, "anomalies": anomalies,
        "risks": risks, "recommendations": recs, "confidence": confidence,
        "source": "rules", "model": None, "grounding": {"numbers_checked": 0, "unverified": []},
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def _compact(facts: dict) -> str:
    """Fact sheet as dense JSON: free-tier token limits make every character count."""
    return json.dumps(facts, separators=(",", ":"))


def _texts(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _texts(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in _texts(v)]
    return []


def generate_insights(facts: dict, use_llm: bool = True) -> dict:
    """Causal summary, anomaly interpretation, risks and recommendations for one horizon."""
    fallback = rule_based_insights(facts)
    if not use_llm or not available_providers():
        fallback["note"] = "No LLM key configured: written by rules from the same fact sheet."
        return fallback

    reply = call_llm(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": f"FACT SHEET:\n{_compact(facts)}\n\n{INSIGHT_SCHEMA}"}],
        json_mode=True)
    parsed = _parse_json(reply.get("text", "")) if "text" in reply else None
    required = {"headline", "executive_summary", "causal_drivers", "risks", "recommendations"}
    if not parsed or not required <= set(parsed):
        fallback["note"] = f"LLM unavailable ({reply.get('error', 'unparseable response')}): written by rules."
        return fallback

    for key in ("causal_drivers", "anomalies", "risks", "recommendations"):
        parsed[key] = [x for x in parsed.get(key) or [] if isinstance(x, dict)]
    parsed.setdefault("confidence", fallback["confidence"])
    parsed["source"] = f"llm:{reply['provider']}"
    parsed["model"] = reply["model"]
    parsed["grounding"] = check_grounding(_texts({k: parsed[k] for k in parsed if k not in ("source", "model")}), facts)
    return parsed


def chat(question: str, facts: dict, history: list[dict] | None = None) -> dict:
    """Free-form question about the current forecast / scenario, answered from the fact sheet."""
    if not available_providers():
        return {"answer": "No LLM key is configured. Add GROQ_API_KEY (or GEMINI_API_KEY / OPENAI_API_KEY) to .env "
                          "to ask questions; the Insights panel above is written from the same facts.",
                "source": "rules", "grounding": {"numbers_checked": 0, "unverified": []}}
    messages = [{"role": "system", "content": SYSTEM_PROMPT
                 + "\nAnswer the user's question in at most 150 words. If the fact sheet cannot answer it, say what "
                   "is missing and which control in the dashboard (budget simulator, optimiser, horizon) would."
                 + f"\n\nFACT SHEET:\n{_compact(facts)}"}]
    messages += [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:]]
    messages.append({"role": "user", "content": question})
    reply = call_llm(messages, max_tokens=700, temperature=0.3)
    if "text" not in reply:
        return {"answer": f"The LLM could not be reached ({reply.get('error')}).", "source": "error",
                "grounding": {"numbers_checked": 0, "unverified": []}}
    return {"answer": reply["text"], "source": f"llm:{reply['provider']}", "model": reply["model"],
            "grounding": check_grounding([reply["text"]], facts)}
