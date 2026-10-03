"""Every chart in the dashboard, as plain Plotly figures (no Streamlit in here).

Visual rules, applied throughout:
- colour follows the entity: one fixed, colour-blind-safe hue per channel; grey for
  context series; blue = the forecast, orange = the user's plan, red = a decrease
- one y-axis per chart; thin marks; solid hairline grids; text in ink, never in a
  series colour; a legend whenever there are two or more series
- ranges are drawn as a light band, medians as a 2px line
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import explain, scenario
from src.config import INTERVAL_COVERAGE, MATURITY_LAG_DAYS
from src.hierarchy import Hierarchy

# Two palettes with the same roles. Each mode's channel hues were validated as a set
# for colour-blind separation against that mode's surface; dark is not an inverted light.
_THEMES = {
    "light": dict(
        CHANNEL_COLOR={"google": "#2a78d6", "meta": "#eb6834", "bing": "#1baf7a", "all": "#52514e"},
        ACCENT="#2a78d6", ACCENT_SOFT="rgba(42,120,214,0.12)", ACCENT_MID="rgba(42,120,214,0.22)",
        ACCENT_DEEP="#104281", ALT="#eb6834", DOWN="#e34948",
        INK="#0b0b0b", INK_2="#52514e", MUTED="#898781", GRID="#e1e0d9", AXIS="#c3c2b7", DEEMPH="#c3c2b7",
        CONTEXT=["#dcdbd3", "#a9a79c", "#74726a"],       # context series, weakest to strongest
        SURFACE="#ffffff", HOVER_BG="#ffffff", TEMPLATE="plotly_white",
    ),
    "dark": dict(
        CHANNEL_COLOR={"google": "#3987e5", "meta": "#d95926", "bing": "#199e70", "all": "#c3c2b7"},
        ACCENT="#3987e5", ACCENT_SOFT="rgba(57,135,229,0.16)", ACCENT_MID="rgba(57,135,229,0.30)",
        ACCENT_DEEP="#9ec5f4", ALT="#e9793f", DOWN="#e66767",
        INK="#ffffff", INK_2="#c3c2b7", MUTED="#8f8d84", GRID="#2b2d33", AXIS="#454850", DEEMPH="#5d6068",
        CONTEXT=["#3b3e46", "#5d6068", "#8f8d84"],
        SURFACE="#171a20", HOVER_BG="#242831", TEMPLATE="plotly_dark",
    ),
}
CHANNEL_COLOR = ACCENT = ACCENT_SOFT = ACCENT_MID = ACCENT_DEEP = ALT = DOWN = None
INK = INK_2 = MUTED = GRID = AXIS = DEEMPH = CONTEXT = SURFACE = HOVER_BG = TEMPLATE = None


def use_theme(dark: bool = False) -> None:
    """Switch every chart drawn afterwards to the light or the dark palette."""
    globals().update(_THEMES["dark" if dark else "light"])


use_theme(False)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LEVEL_LABEL = {"blended": "Blended", "channel": "Channel", "campaign_type": "Campaign type", "campaign": "Campaign"}


def usd(x: float, compact: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    if not compact or abs(x) < 1_000:
        return f"${x:,.0f}"
    if abs(x) < 1_000_000:
        return f"${x / 1_000:,.1f}k" if abs(x) < 100_000 else f"${x / 1_000:,.0f}k"
    return f"${x / 1_000_000:,.2f}M"


def md(text: object) -> str:
    """Escape dollar signs so Streamlit's markdown does not read '$..$' as LaTeX."""
    return str(text).replace("$", "\\$")


def series_label(r: pd.Series) -> str:
    if r["level"] == "blended":
        return "All channels"
    if r["level"] == "channel":
        return r["channel"].title()
    if r["level"] == "campaign_type":
        return f"{r['channel'].title()} · {r['campaign_type'].replace('_', ' ').title()}"
    return f"{r['channel'].title()} · {r['campaign_name']}"


def base_layout(fig: go.Figure, height: int = 330, title: str | None = None, legend: bool = True,
                hovermode: str | bool = "closest") -> go.Figure:
    fig.update_layout(
        template=TEMPLATE, height=height, hovermode=hovermode,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK_2),
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left") if title else None,
        margin=dict(l=8, r=16, t=48 if title else 16, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=HOVER_BG, font=dict(family=FONT, color=INK), bordercolor=AXIS),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED),
                     zeroline=False)
    return fig


# ─── Charts ──────────────────────────────────────────────────────────────────

def history_forecast_chart(hier: Hierarchy, row: pd.Series, horizon: int, scen_row: pd.Series | None = None,
                           weeks: int = 60) -> go.Figure:
    """Weekly revenue history and the average weekly rate implied by the horizon forecast."""
    hist = explain.weekly_history(hier, int(row["series_id"]), weeks)
    if MATURITY_LAG_DAYS:
        hist = hist.iloc[:-1]          # the last week is still collecting conversions
    origin = hier.origin
    end = origin + pd.Timedelta(days=horizon)
    per_week = 7.0 / horizon
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["week_end"], y=hist["revenue"], mode="lines", name="Weekly revenue (actual)",
                             line=dict(color=INK_2, width=2),
                             hovertemplate="Week ending %{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>"))
    lo, mid, hi = (row[f"revenue_{q}"] * per_week for q in ("p10", "p50", "p90"))
    fig.add_trace(go.Scatter(x=[origin, end, end, origin], y=[lo, lo, hi, hi], fill="toself", mode="lines",
                             line=dict(width=0), fillcolor=ACCENT_SOFT, name="P10-P90 range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[origin, end], y=[mid, mid], mode="lines", name="Forecast P50 (weekly average)",
                             line=dict(color=ACCENT, width=2),
                             hovertemplate=f"Next {horizon} days, weekly average<br>P50 $%{{y:,.0f}}<extra></extra>"))
    if scen_row is not None:
        s_mid = scen_row["revenue_p50"] * per_week
        fig.add_trace(go.Scatter(x=[origin, end], y=[s_mid, s_mid], mode="lines", name="Budget scenario P50",
                                 line=dict(color=ALT, width=2),
                                 hovertemplate="Scenario weekly average<br>$%{y:,.0f}<extra></extra>"))
    fig.add_annotation(x=end, y=mid, text=f" P50 {usd(mid)}/wk", showarrow=False, xanchor="left", yanchor="middle",
                       font=dict(color=INK_2, size=12))
    base_layout(fig, 360, hovermode="x")
    fig.update_layout(margin=dict(l=8, r=96, t=16, b=8))
    fig.update_xaxes(range=[hist["week_end"].iloc[0], end])
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig


def waterfall_chart(row: pd.Series) -> go.Figure:
    wf = explain.driver_waterfall(row)
    measure = ["absolute"] + ["relative"] * (len(wf) - 2) + ["total"]
    text = [usd(v) if k != "step" else f"{'+' if v >= 0 else '-'}{usd(abs(v))}"
            for v, k in zip(wf["effect"], wf["kind"])]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=measure, y=wf["short"], x=wf["effect"], text=text, textposition="outside",
        textfont=dict(color=INK_2), connector=dict(line=dict(color=AXIS, width=1)),
        increasing=dict(marker=dict(color=ACCENT)), decreasing=dict(marker=dict(color=DOWN)),
        totals=dict(marker=dict(color=INK_2)), width=0.5, cliponaxis=False,
        customdata=np.stack([wf["driver"], wf["pct"]], axis=1),
        hovertemplate="%{customdata[0]}<br>%{text} (%{customdata[1]:+.1f}%)<extra></extra>"))
    base_layout(fig, max(250, 44 * len(wf) + 50), legend=False)
    fig.update_layout(margin=dict(l=8, r=70, t=16, b=8))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_2))
    running = np.concatenate([[wf["effect"].iloc[0]], wf["effect"].iloc[0] + np.cumsum(wf["effect"].iloc[1:-1])])
    top = float(max(running.max(), wf["effect"].iloc[-1]))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix="$", tickformat="~s", nticks=4, tickangle=0,
                     range=[0, top * 1.05])      # bars keep a zero baseline
    return fig


def channel_bars(fc_h: pd.DataFrame, compare: pd.DataFrame | None = None) -> go.Figure:
    ch = fc_h[fc_h["level"] == "channel"].sort_values("revenue_p50")
    fig = go.Figure()
    if compare is not None:
        base = compare[compare["level"] == "channel"].set_index("channel").reindex(ch["channel"])
        fig.add_trace(go.Bar(y=ch["channel"].str.title(), x=base["revenue_p50"], orientation="h", name="Baseline",
                             marker=dict(color=DEEMPH, cornerradius=4), width=0.3, offsetgroup="a",
                             hovertemplate="%{y} baseline<br>P50 $%{x:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        y=ch["channel"].str.title(), x=ch["revenue_p50"], orientation="h",
        name="Your plan" if compare is not None else "P50",
        marker=dict(color=ACCENT if compare is not None else [CHANNEL_COLOR.get(c, INK_2) for c in ch["channel"]],
                    cornerradius=4),
        width=0.3, offsetgroup="b",
        error_x=dict(type="data", symmetric=False, array=ch["revenue_p90"] - ch["revenue_p50"],
                     arrayminus=ch["revenue_p50"] - ch["revenue_p10"], color=INK_2, thickness=1.2, width=4),
        customdata=np.stack([ch["revenue_p10"], ch["revenue_p90"], ch["roas_p50"]], axis=1),
        hovertemplate="%{y}<br>P50 $%{x:,.0f}<br>P10-P90 $%{customdata[0]:,.0f} - $%{customdata[1]:,.0f}"
                      "<br>ROAS %{customdata[2]:.2f}<extra></extra>"))
    base_layout(fig, 90 + 62 * len(ch), legend=compare is not None)
    fig.update_layout(bargap=0.35, barmode="group")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix="$", tickformat="~s", rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def roas_range_chart(fc_h: pd.DataFrame, breakeven: float = 1.0) -> go.Figure:
    t = fc_h[(fc_h["level"] == "campaign_type") & (fc_h["spend_p50"] > 0)].copy()
    t = t[t["spend_p50"] >= 0.01 * t["spend_p50"].sum()]          # tiny lines have meaningless ROAS ranges
    t["label"] = t.apply(series_label, axis=1)
    t = t.sort_values("roas_p50")
    fig = go.Figure()
    for channel in [c for c in CHANNEL_COLOR if c in set(t["channel"])]:
        g = t[t["channel"] == channel]
        color = CHANNEL_COLOR[channel]
        for _, r in g.iterrows():
            fig.add_trace(go.Scatter(x=[r["roas_p10"], r["roas_p90"]], y=[r["label"], r["label"]], mode="lines",
                                     line=dict(color=color, width=2), opacity=0.45, showlegend=False,
                                     hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=g["roas_p50"], y=g["label"], mode="markers", name=channel.title(),
            marker=dict(color=color, size=11, line=dict(color=SURFACE, width=2)),
            customdata=np.stack([g["roas_p10"], g["roas_p90"], g["spend_p50"]], axis=1),
            hovertemplate="%{y}<br>ROAS P50 %{x:.2f}<br>P10-P90 %{customdata[0]:.2f} - %{customdata[1]:.2f}"
                          "<br>Spend $%{customdata[2]:,.0f}<extra></extra>"))
    fig.add_vline(x=breakeven, line=dict(color=INK_2, width=1))
    fig.add_annotation(x=breakeven, y=0, yref="paper", text=f" break-even {breakeven:g}x", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=MUTED, size=11))
    base_layout(fig, 120 + 36 * len(t))
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=t["label"].tolist()))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, ticksuffix="x", range=[0, float(t["roas_p90"].max()) * 1.08])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def response_charts(fc: pd.DataFrame, scen: pd.DataFrame, series_id: int, horizon: int) -> tuple[go.Figure, go.Figure]:
    curve = scenario.response_curve(fc, series_id, horizon)
    base = fc[(fc["series_id"] == series_id) & (fc["horizon_days"] == horizon)].iloc[0]
    plan = scen[(scen["series_id"] == series_id) & (scen["horizon_days"] == horizon)].iloc[0]

    rev = go.Figure()
    rev.add_trace(go.Scatter(x=list(curve["spend"]) + list(curve["spend"][::-1]),
                             y=list(curve["revenue_p90"]) + list(curve["revenue_p10"][::-1]), fill="toself",
                             mode="lines", line=dict(width=0), fillcolor=ACCENT_SOFT, name="P10-P90 range",
                             hoverinfo="skip"))
    rev.add_trace(go.Scatter(x=curve["spend"], y=curve["revenue_p50"], mode="lines", name="Expected revenue",
                             line=dict(color=ACCENT, width=2),
                             hovertemplate="Spend $%{x:,.0f}<br>Revenue $%{y:,.0f}<extra></extra>"))
    rev.add_trace(go.Scatter(x=[base["spend_p50"]], y=[base["revenue_p50"]], mode="markers", name="Expected spend",
                             marker=dict(color=INK_2, size=11, line=dict(color=SURFACE, width=2)),
                             hovertemplate="Expected<br>Spend $%{x:,.0f}<br>Revenue $%{y:,.0f}<extra></extra>"))
    rev.add_trace(go.Scatter(x=[plan["spend_p50"]], y=[plan["revenue_p50"]], mode="markers", name="Your plan",
                             marker=dict(color=ALT, size=11, line=dict(color=SURFACE, width=2)),
                             hovertemplate="Plan<br>Spend $%{x:,.0f}<br>Revenue $%{y:,.0f}<extra></extra>"))
    base_layout(rev, 320)
    rev.update_xaxes(tickprefix="$", tickformat="~s", title=dict(text=f"Spend over {horizon} days", font=dict(color=MUTED)))
    rev.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")

    mar = go.Figure()
    mar.add_trace(go.Scatter(x=curve["spend"], y=curve["marginal_roas"], mode="lines", name="Marginal ROAS",
                             line=dict(color=ACCENT, width=2),
                             hovertemplate="Spend $%{x:,.0f}<br>Next dollar returns $%{y:.2f}<extra></extra>"))
    mar.add_trace(go.Scatter(x=curve["spend"], y=curve["roas"], mode="lines", name="Average ROAS",
                             line=dict(color=DEEMPH, width=2),
                             hovertemplate="Spend $%{x:,.0f}<br>Average ROAS %{y:.2f}<extra></extra>"))
    base_layout(mar, 320)
    mar.update_xaxes(tickprefix="$", tickformat="~s", title=dict(text=f"Spend over {horizon} days", font=dict(color=MUTED)))
    mar.update_yaxes(ticksuffix="x", rangemode="tozero")
    return rev, mar


def allocation_chart(alloc: pd.DataFrame) -> go.Figure:
    a = alloc.copy()
    a["label"] = a["channel"].str.title() + " · " + a["campaign_type"].str.replace("_", " ").str.title()
    a = a.sort_values("spend_recommended")
    fig = go.Figure()
    for _, r in a.iterrows():
        fig.add_trace(go.Scatter(x=[r["spend_current"], r["spend_recommended"]], y=[r["label"], r["label"]],
                                 mode="lines", line=dict(color=AXIS, width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a["spend_current"], y=a["label"], mode="markers", name="Expected spend",
                             marker=dict(color=DEEMPH, size=11, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y}<br>Expected $%{x:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a["spend_recommended"], y=a["label"], mode="markers", name="Recommended",
                             marker=dict(color=ACCENT, size=11, line=dict(color=SURFACE, width=2)),
                             customdata=np.stack([a["change_pct"], a["marginal_roas_recommended"]], axis=1),
                             hovertemplate="%{y}<br>Recommended $%{x:,.0f} (%{customdata[0]:+.0f}%)"
                                           "<br>Marginal ROAS %{customdata[1]:.2f}<extra></extra>"))
    base_layout(fig, 110 + 34 * len(a))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix="$", tickformat="~s", rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def accuracy_chart(bt: pd.DataFrame, horizon: int) -> go.Figure:
    b = bt[bt["horizon_days"] == horizon].copy()
    b["label"] = b["level"].map(LEVEL_LABEL)
    order = [LEVEL_LABEL[l] for l in ("campaign", "campaign_type", "channel", "blended")]
    series = [("wape_run_rate", "Run-rate baseline", CONTEXT[0]), ("wape_seasonal_naive", "Seasonal-naive baseline", CONTEXT[1]),
              ("wape", "AIgnition v4", ACCENT), ("wape_budget_known", "AIgnition v4, budget known", ACCENT_DEEP)]
    fig = go.Figure()
    for col, name, color in series:
        fig.add_trace(go.Bar(y=b["label"], x=b[col], orientation="h", name=name,
                             marker=dict(color=color, cornerradius=3), text=[f"{v:.0f}%" for v in b[col]],
                             textposition="outside", textfont=dict(color=INK_2, size=11),
                             hovertemplate=f"{name}<br>%{{y}}: %{{x:.1f}}% WAPE<extra></extra>"))
    base_layout(fig, 400)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.12,
                      yaxis=dict(categoryorder="array", categoryarray=order))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def legacy_chart(legacy: pd.DataFrame, horizon: int) -> go.Figure:
    g = legacy[legacy["horizon_days"] == horizon].copy()
    g["label"] = g["level"].map(LEVEL_LABEL)
    order = [LEVEL_LABEL[l] for l in ("campaign", "campaign_type", "channel", "blended")]
    fig = go.Figure()
    for _, r in g.iterrows():
        fig.add_trace(go.Scatter(x=[r["wape_v31"], r["wape_v4"]], y=[r["label"], r["label"]], mode="lines",
                                 line=dict(color=AXIS, width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["wape_v31"], y=g["label"], mode="markers+text", name="Previous pipeline (v3.1)",
                             text=[f"{v:.0f}%" for v in g["wape_v31"]], textposition="top center",
                             textfont=dict(color=MUTED, size=11),
                             marker=dict(color=DEEMPH, size=12, line=dict(color=SURFACE, width=2)),
                             hovertemplate="v3.1 %{y}: %{x:.1f}% WAPE<extra></extra>"))
    fig.add_trace(go.Scatter(x=g["wape_v4"], y=g["label"], mode="markers+text", name="AIgnition v4",
                             text=[f"{v:.0f}%" for v in g["wape_v4"]], textposition="top center",
                             textfont=dict(color=INK_2, size=11),
                             marker=dict(color=ACCENT, size=12, line=dict(color=SURFACE, width=2)),
                             hovertemplate="v4 %{y}: %{x:.1f}% WAPE<extra></extra>"))
    base_layout(fig, 300)
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=order))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def coverage_chart(bt: pd.DataFrame) -> go.Figure:
    b = bt.copy()
    b["label"] = b["level"].map(LEVEL_LABEL) + " · " + b["horizon_days"].astype(str) + "d"
    b = b.iloc[::-1]
    target = INTERVAL_COVERAGE * 100
    fig = go.Figure(go.Bar(y=b["label"], x=b["coverage"], orientation="h",
                           marker=dict(color=ACCENT, cornerradius=4), width=0.55,
                           text=[f"{v:.0f}%" for v in b["coverage"]], textposition="outside",
                           textfont=dict(color=INK_2, size=11), cliponaxis=False,
                           hovertemplate="%{y}<br>Actual inside P10-P90: %{x:.1f}%<extra></extra>"))
    fig.add_vline(x=target, line=dict(color=INK_2, width=1))
    fig.add_annotation(x=target, y=1.0, yref="paper", text=f"target {target:.0f}%", showarrow=False,
                       xanchor="center", yanchor="bottom", font=dict(color=INK_2, size=11))
    base_layout(fig, 70 + 26 * len(b), legend=False)
    fig.update_layout(margin=dict(l=8, r=40, t=28, b=8))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, ticksuffix="%", range=[0, 105])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def replay_chart(replay: pd.DataFrame, horizon: int, channel: str = "all") -> go.Figure:
    """Every past forecast of one top-level series against what then happened."""
    r = replay[(replay["channel"] == channel) & (replay["horizon_days"] == horizon)].sort_values("origin")
    x = pd.to_datetime(r["origin"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(r["revenue_p90"]) + list(r["revenue_p10"][::-1]),
                             fill="toself", mode="lines", line=dict(width=0), fillcolor=ACCENT_SOFT,
                             name="P10-P90 issued", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=r["actual_revenue"], mode="lines", name="What happened",
                             line=dict(color=INK, width=2), hovertemplate="Actual $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=r["revenue_p50"], mode="lines", name="Forecast P50",
                             line=dict(color=ACCENT, width=2), hovertemplate="Forecast $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=r["budget_revenue_p50"], mode="lines", name="Forecast P50, budget known",
                             line=dict(color=ALT, width=2), hovertemplate="Budget known $%{y:,.0f}<extra></extra>"))
    base_layout(fig, 360, hovermode="x unified")
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    fig.update_xaxes(title=dict(text="Date the forecast was issued", font=dict(color=MUTED)))
    return fig


def small_multiples_history(hier: Hierarchy, weeks: int = 104) -> go.Figure:
    channels = hier.meta[hier.meta["level"] == "channel"]
    fig = make_subplots(rows=1, cols=len(channels), subplot_titles=[c.title() for c in channels["channel"]])
    for i, (_, m) in enumerate(channels.iterrows(), 1):
        wk = explain.weekly_history(hier, int(m["series_id"]), weeks)
        fig.add_trace(go.Scatter(x=wk["week_end"], y=wk["revenue"], mode="lines", name="Revenue",
                                 line=dict(color=CHANNEL_COLOR.get(m["channel"], INK_2), width=2),
                                 showlegend=False, hovertemplate="%{x|%b %d, %Y}<br>Revenue $%{y:,.0f}<extra></extra>"),
                      row=1, col=i)
        fig.add_trace(go.Scatter(x=wk["week_end"], y=wk["spend"], mode="lines", name="Spend",
                                 line=dict(color=DEEMPH, width=2), showlegend=False,
                                 hovertemplate="%{x|%b %d, %Y}<br>Spend $%{y:,.0f}<extra></extra>"), row=1, col=i)
    base_layout(fig, 290, legend=False, hovermode="x")
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    fig.update_annotations(font=dict(color=INK_2, size=13))
    return fig


def cone_chart(rows: pd.DataFrame, scen_rows: pd.DataFrame | None = None) -> go.Figure:
    """Cumulative revenue over the planning window as a widening cone.

    `rows` are one series' forecast rows (one per horizon).  The outer band is
    P10-P90, the inner band P25-P75 (from the split-normal through the three
    forecast quantiles); the grey line is what the last-28-day pace would give.
    """
    r = rows.sort_values("horizon_days")
    x = [0] + r["horizon_days"].tolist()
    q = {name: [0.0] + [scenario.quantile_from_three(a, b, c, prob)
                        for a, b, c in zip(r["revenue_p10"], r["revenue_p50"], r["revenue_p90"])]
         for name, prob in (("p10", 0.10), ("p25", 0.25), ("p75", 0.75), ("p90", 0.90))}
    p50 = [0.0] + r["revenue_p50"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=q["p90"] + q["p10"][::-1], fill="toself", mode="lines",
                             line=dict(width=0), fillcolor=ACCENT_SOFT, name="P10-P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x + x[::-1], y=q["p75"] + q["p25"][::-1], fill="toself", mode="lines",
                             line=dict(width=0), fillcolor=ACCENT_MID, name="P25-P75", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[0.0] + r["baseline_revenue"].tolist(), mode="lines", name="At the 28-day pace",
                             line=dict(color=DEEMPH, width=2),
                             hovertemplate="Run-rate $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode="lines+markers", name="Forecast P50", line=dict(color=ACCENT, width=2),
        marker=dict(size=9, color=ACCENT, line=dict(color=SURFACE, width=2)),
        customdata=np.stack([q["p10"], q["p90"]], axis=1),
        hovertemplate="P50 $%{y:,.0f} (P10-P90 $%{customdata[0]:,.0f} - $%{customdata[1]:,.0f})<extra></extra>"))
    if scen_rows is not None:
        sr = scen_rows.sort_values("horizon_days")
        fig.add_trace(go.Scatter(x=x, y=[0.0] + sr["revenue_p50"].tolist(), mode="lines+markers", name="Your plan P50",
                                 line=dict(color=ALT, width=2),
                                 marker=dict(size=9, color=ALT, line=dict(color=SURFACE, width=2)),
                                 hovertemplate="Plan $%{y:,.0f}<extra></extra>"))
    for day, value in zip(x[1:], p50[1:]):
        fig.add_annotation(x=day, y=value, text=usd(value), showarrow=False, yshift=14, xanchor="right",
                           font=dict(color=INK_2, size=12))
    base_layout(fig, 360, hovermode="x unified")
    fig.update_xaxes(tickvals=x, ticksuffix="d", title=dict(text="Days after the forecast origin", font=dict(color=MUTED)))
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig


def exceedance_chart(p10: float, p50: float, p90: float, target: float) -> go.Figure:
    """Chance of reaching at least each revenue level, with the goal marked."""
    curve = scenario.outcome_curve(p10, p50, p90)
    chance = 100 - curve["percentile"]
    prob = scenario.probability_at_least(p10, p50, p90, target) * 100
    fig = go.Figure()
    reached = curve["value"] >= target
    if reached.any():
        fig.add_trace(go.Scatter(x=curve["value"][reached], y=chance[reached], mode="lines", fill="tozeroy",
                                 line=dict(width=0), fillcolor=ACCENT_SOFT, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=curve["value"], y=chance, mode="lines", line=dict(color=ACCENT, width=2),
                             hovertemplate="$%{x:,.0f} or more: %{y:.0f}% chance<extra></extra>", showlegend=False))
    fig.add_vline(x=target, line=dict(color=INK_2, width=1))
    fig.add_annotation(x=target, y=1.0, yref="paper", text=f" goal {usd(target)}: {prob:.0f}%", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=INK, size=12))
    base_layout(fig, 250, legend=False, hovermode="x")
    fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=False)
    fig.update_yaxes(ticksuffix="%", range=[0, 102], title=dict(text="Chance of at least", font=dict(color=MUTED)))
    return fig


def seasonality_chart(hier: Hierarchy, series_id: int, horizon: int, metric: str = "revenue") -> go.Figure:
    """Weekly values by week of the year, one line per year, with the forecast window shaded."""
    data = explain.yearly_overlay(hier, series_id)
    years = sorted(data["year"].unique())
    shades = CONTEXT
    past = years[:-1][-len(shades):]
    fig = go.Figure()
    for year in years:
        d = data[data["year"] == year]
        if year == years[-1]:
            color = ACCENT
        elif year in past:
            color = shades[len(shades) - len(past) + past.index(year)]
        else:
            continue
        fig.add_trace(go.Scatter(
            x=d["week"], y=d[metric], mode="lines", name=str(year), line=dict(color=color, width=2),
            customdata=d["week_end"].dt.strftime("%b %d, %Y"),
            hovertemplate=f"{year}, week ending %{{customdata}}: $%{{y:,.0f}}<extra></extra>"))
    start = int((hier.origin + pd.Timedelta(days=1)).isocalendar().week)
    end = int((hier.origin + pd.Timedelta(days=horizon)).isocalendar().week)
    spans = [(start, end)] if end >= start else [(start, 53), (1, end)]
    for a, b in spans:
        fig.add_vrect(x0=a - 0.5, x1=b + 0.5, fillcolor=ACCENT_SOFT, line_width=0, layer="below")
    fig.add_annotation(x=spans[0][0] - 0.5, y=1.0, yref="paper", text=f" next {horizon} days", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=INK_2, size=11))
    base_layout(fig, 300, hovermode="x unified")
    fig.update_xaxes(title=dict(text="Week of the year", font=dict(color=MUTED)), range=[0.5, 53.5], dtick=4)
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig


def movers_chart(mv: pd.DataFrame) -> go.Figure:
    """Largest departures from the run-rate, in dollars: blue up, red down, biggest at the top."""
    m = mv.copy()
    m["label"] = m.apply(series_label, axis=1)
    m = m.reindex(m["change_usd"].abs().sort_values().index)
    fig = go.Figure(go.Bar(
        y=m["label"], x=m["change_usd"], orientation="h", width=0.5,
        marker=dict(color=[ACCENT if v >= 0 else DOWN for v in m["change_usd"]], cornerradius=4),
        customdata=np.stack([m["baseline_revenue"], m["revenue_p50"], m["change_pct"].fillna(0)], axis=1),
        hovertemplate="%{y}<br>Run-rate $%{customdata[0]:,.0f}, forecast $%{customdata[1]:,.0f}"
                      " (%{customdata[2]:+.0f}%)<extra></extra>"))
    # value labels sit on the empty side of the zero line, so they never collide with a bar or a name
    for label, v in zip(m["label"], m["change_usd"]):
        fig.add_annotation(x=0, y=label, text=f"{'+' if v >= 0 else '-'}{usd(abs(v))}", showarrow=False,
                           xanchor="right" if v >= 0 else "left", xshift=-8 if v >= 0 else 8,
                           font=dict(color=INK_2, size=12))
    span = float(m["change_usd"].abs().max()) if len(m) else 1.0
    base_layout(fig, 90 + 40 * len(m), legend=False)
    fig.update_layout(margin=dict(l=8, r=20, t=8, b=8))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix="$", tickformat="~s", nticks=5,
                     range=[-span * 1.15, span * 1.15])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def scenarios_chart(saved: pd.DataFrame) -> go.Figure:
    """Saved budget plans side by side: P50 revenue with its P10-P90 range."""
    d = saved.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=d["Plan"], x=d["Revenue P50"], orientation="h", width=0.45,
        marker=dict(color=[DEEMPH if name == "Baseline" else ACCENT for name in d["Plan"]], cornerradius=4),
        error_x=dict(type="data", symmetric=False, array=d["Revenue P90"] - d["Revenue P50"],
                     arrayminus=d["Revenue P50"] - d["Revenue P10"], color=INK_2, thickness=1.2, width=4),
        customdata=np.stack([d["Spend"], d["ROAS"]], axis=1),
        hovertemplate="%{y}<br>Revenue P50 $%{x:,.0f}<br>Spend $%{customdata[0]:,.0f}"
                      "<br>ROAS %{customdata[1]:.2f}x<extra></extra>"))
    base_layout(fig, 80 + 46 * len(d), legend=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix="$", tickformat="~s", rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig
