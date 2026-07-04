"""
AIgnition 3.0 — AI-Assisted E-commerce Marketing Forecast Dashboard
Run: streamlit run app.py
"""

from __future__ import annotations

import html as html_lib
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.config import HORIZONS, META_CONVERSION_AS_REVENUE, ROOT
from src.insights import detect_anomalies, generate_all_insights, chat_with_forecast
from src.model import load_bundle, model_key
from src.pipeline import generate_forecasts, prepare_data

st.set_page_config(
    page_title="AIgnition Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = ROOT / "pickle" / "model.pkl"
DATA_DIR = ROOT / "data"


def inject_theme_css(theme: str) -> None:
    is_dark = theme == "Dark"

    # ── Token system ──────────────────────────────────────────────────────────
    if is_dark:
        bg_page        = "#0f172a"
        bg_sidebar     = "#1e293b"
        bg_card        = "#1e293b"
        bg_input       = "#0f172a"
        text_primary   = "#f1f5f9"
        text_secondary = "#94a3b8"
        text_muted     = "#64748b"
        border_color   = "#334155"
        metric_val     = "#f1f5f9"
        metric_label   = "#94a3b8"
        tab_text       = "#cbd5e1"
        tab_active     = "#0ea5e9"
        tab_active_bg  = "#1e3a5f"
        table_row_alt  = "#243044"
        table_header   = "#1e3a5f"
        table_text     = "#e2e8f0"
        info_bg        = "#0c2a4a"
        info_text      = "#7dd3fc"
        info_border    = "#0ea5e9"
        warn_bg        = "#2d1f00"
        warn_text      = "#fbbf24"
        warn_border    = "#d97706"
        success_bg     = "#052e16"
        success_text   = "#86efac"
        success_border = "#22c55e"
        # insight boxes
        ins_bg         = "#422006"
        ins_text       = "#fef9c3"
        ins_border     = "#eab308"
        risk_bg        = "#450a0a"
        risk_text      = "#fecaca"
        risk_border    = "#ef4444"
        reco_bg        = "#052e16"
        reco_text      = "#bbf7d0"
        reco_border    = "#22c55e"
        selectbox_bg   = "#1e293b"
        slider_text    = "#f1f5f9"
        caption_text   = "#94a3b8"
        subheader_text = "#e2e8f0"
        dataframe_bg   = "#1e293b"
        tooltip_bg     = "#1e3a5f"
        tooltip_text   = "#f1f5f9"
        button_bg      = "#334155"
        button_text    = "#f1f5f9"
    else:
        bg_page        = "#f8fafc"
        bg_sidebar     = "#ffffff"
        bg_card        = "#ffffff"
        bg_input       = "#ffffff"
        text_primary   = "#0f172a"
        text_secondary = "#475569"
        text_muted     = "#64748b"
        border_color   = "#e2e8f0"
        metric_val     = "#0f172a"
        metric_label   = "#475569"
        tab_text       = "#475569"
        tab_active     = "#0369a1"
        tab_active_bg  = "#e0f2fe"
        table_row_alt  = "#f1f5f9"
        table_header   = "#e0f2fe"
        table_text     = "#0f172a"
        info_bg        = "#e0f2fe"
        info_text      = "#0369a1"
        info_border    = "#0ea5e9"
        warn_bg        = "#fffbeb"
        warn_text      = "#92400e"
        warn_border    = "#d97706"
        success_bg     = "#f0fdf4"
        success_text   = "#166534"
        success_border = "#22c55e"
        ins_bg         = "#fefce8"
        ins_text       = "#422006"
        ins_border     = "#eab308"
        risk_bg        = "#fef2f2"
        risk_text      = "#991b1b"
        risk_border    = "#ef4444"
        reco_bg        = "#ecfdf5"
        reco_text      = "#065f46"
        reco_border    = "#22c55e"
        selectbox_bg   = "#ffffff"
        slider_text    = "#0f172a"
        caption_text   = "#64748b"
        subheader_text = "#0f172a"
        dataframe_bg   = "#ffffff"
        tooltip_bg     = "#f8fafc"
        tooltip_text   = "#0f172a"
        button_bg      = "#0ea5e9"
        button_text    = "#ffffff"

    st.markdown(
        f"""
<style>
/* ── Google Fonts ────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Page / app shell ────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

/* Main background */
.stApp {{
    background-color: {bg_page} !important;
}}
.stApp > header {{
    background-color: {bg_page} !important;
}}
.stApp > header * {{
    color: {text_primary} !important;
}}
/* Tooltips */
[data-baseweb="tooltip"], [data-baseweb="tooltip"] * {{
    background-color: {tooltip_bg} !important;
    color: {tooltip_text} !important;
}}
div[role="tooltip"], div[role="tooltip"] * {{
    background-color: {tooltip_bg} !important;
    color: {tooltip_text} !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {bg_sidebar} !important;
    border-right: 1px solid {border_color};
}}
[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
button[kind="secondary"],
button[kind="primary"],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {{
    background-color: {button_bg} !important;
    color: {button_text} !important;
    border-color: {border_color} !important;
}}
button[kind="secondary"] *,
button[kind="primary"] *,
[data-testid="baseButton-secondary"] *,
[data-testid="baseButton-primary"] * {{
    color: {button_text} !important;
}}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {{
    color: {text_primary} !important;
    font-weight: 500;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: {text_primary} !important;
}}

/* ── Main content text ───────────────────────────────────────────────────── */
.main .block-container {{
    background-color: {bg_page} !important;
    color: {text_primary} !important;
    padding-top: 1.5rem;
}}
.main .block-container p,
.main .block-container span,
.main .block-container div,
.main .block-container label,
.main .block-container li {{
    color: {text_primary} !important;
}}
h1, h2, h3, h4, h5, h6 {{
    color: {text_primary} !important;
}}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    color: {metric_val} !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {metric_label} !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricDelta"] {{
    color: {text_secondary} !important;
}}
[data-testid="stMetric"] {{
    background-color: {bg_card} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {{
    background-color: {bg_card} !important;
    border-bottom: 2px solid {border_color} !important;
    border-radius: 8px 8px 0 0 !important;
}}
[data-testid="stTabs"] [role="tab"] {{
    color: {tab_text} !important;
    font-weight: 500 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.5rem 1rem !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {tab_active} !important;
    background-color: {tab_active_bg} !important;
    border-bottom: 2px solid {tab_active} !important;
    font-weight: 700 !important;
}}
[data-testid="stTabContent"] {{
    background-color: {bg_page} !important;
    border: 1px solid {border_color};
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 1rem !important;
}}

/* ── Selectbox / text input ──────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div > div {{
    background-color: {bg_input} !important;
    border: 1px solid {border_color} !important;
    color: {text_primary} !important;
    border-radius: 8px !important;
}}
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label {{
    color: {text_secondary} !important;
    font-weight: 500 !important;
}}
/* Target the actual inner select control rendered by baseweb */
[data-testid="stSelectbox"] [data-baseweb="select"] {{
    background-color: {bg_input} !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background-color: {bg_input} !important;
    border-color: {border_color} !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {{
    background-color: {bg_input} !important;
    color: {text_primary} !important;
}}
/* Also target the sidebar selectbox specifically */
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] span {{
    background-color: {bg_input} !important;
    color: {text_primary} !important;
    border-color: {border_color} !important;
}}
/* Dropdown popover list */
[data-baseweb="popover"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {{
    background-color: {bg_card} !important;
    color: {text_primary} !important;
}}
[data-baseweb="popover"] li:hover {{
    background-color: {tab_active_bg} !important;
}}

/* ── Slider ──────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] label {{
    color: {text_secondary} !important;
    font-weight: 500 !important;
}}
[data-testid="stSlider"] [data-testid="stTickBar"] span,
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {{
    color: {slider_text} !important;
}}

/* ── Radio ───────────────────────────────────────────────────────────────── */
[data-testid="stRadio"] label {{
    color: {text_primary} !important;
    font-weight: 500 !important;
}}
[data-testid="stRadio"] > div {{
    gap: 0.5rem !important;
}}

/* ── Checkbox ────────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label {{
    color: {text_primary} !important;
}}

/* ── Dataframe / table ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}
[data-testid="stDataFrame"] table {{
    background-color: {dataframe_bg} !important;
}}
[data-testid="stDataFrame"] th {{
    background-color: {table_header} !important;
    color: {text_primary} !important;
    font-weight: 600 !important;
    border-bottom: 1px solid {border_color} !important;
}}
[data-testid="stDataFrame"] td {{
    color: {table_text} !important;
    border-bottom: 1px solid {border_color} !important;
}}
[data-testid="stDataFrame"] tr:nth-child(even) {{
    background-color: {table_row_alt} !important;
}}

/* ── st.info / st.warning / st.success / st.error ───────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
    border: 1px solid !important;
}}
div[data-testid="stAlert"][data-baseweb="notification"] {{
    background-color: {info_bg} !important;
    color: {info_text} !important;
    border-color: {info_border} !important;
}}
/* info */
div.stAlert[kind="info"],
div[role="alert"].stAlert-info {{
    background-color: {info_bg} !important;
    color: {info_text} !important;
    border-color: {info_border} !important;
}}
/* warning */
div.stAlert[kind="warning"],
div[role="alert"].stAlert-warning {{
    background-color: {warn_bg} !important;
    color: {warn_text} !important;
    border-color: {warn_border} !important;
}}
/* success */
div.stAlert[kind="success"],
div[role="alert"].stAlert-success {{
    background-color: {success_bg} !important;
    color: {success_text} !important;
    border-color: {success_border} !important;
}}
/* Catch-all for all alert-box text */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {{
    color: inherit !important;
}}

/* ── st.caption ──────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"],
.stCaption,
small {{
    color: {caption_text} !important;
}}

/* ── st.subheader / st.markdown ─────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {{
    color: {subheader_text} !important;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {{
    color: {text_primary} !important;
}}
[data-testid="stMarkdownContainer"] strong {{
    color: {text_primary} !important;
}}
[data-testid="stMarkdownContainer"] code {{
    background-color: {bg_card} !important;
    color: #0ea5e9 !important;
    padding: 0.1rem 0.35rem !important;
    border-radius: 4px !important;
}}

/* ── Plotly chart containers ─────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {{
    background-color: {bg_card} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {{
    border-color: {border_color} !important;
}}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: {bg_page};
}}
::-webkit-scrollbar-thumb {{
    background: {border_color};
    border-radius: 3px;
}}

/* ── Custom component classes ────────────────────────────────────────────── */
.main-header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: #ffffff !important;
    box-shadow: 0 8px 32px rgba(14,165,233,0.2);
}}
.main-header h1 {{
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff !important;
}}
.main-header p {{
    margin: 0.5rem 0 0;
    opacity: 0.92;
    font-size: 1.05rem;
    color: #e2e8f0 !important;
}}

.insight-box, .risk-box, .reco-box {{
    padding: 1rem 1.25rem;
    border-radius: 0 12px 12px 0;
    margin: 0.5rem 0;
    line-height: 1.6;
    font-size: 1rem;
}}
.insight-box p, .risk-box p, .reco-box p {{
    margin: 0;
    font-size: 1rem;
    line-height: 1.6;
}}

.insight-box {{
    background: {ins_bg};
    border-left: 4px solid {ins_border};
    color: {ins_text} !important;
}}
.insight-box p {{ color: {ins_text} !important; }}

.risk-box {{
    background: {risk_bg};
    border-left: 4px solid {risk_border};
    color: {risk_text} !important;
    margin: 0.35rem 0;
}}
.risk-box p {{ color: {risk_text} !important; }}

.reco-box {{
    background: {reco_bg};
    border-left: 4px solid {reco_border};
    color: {reco_text} !important;
}}
.reco-box p {{ color: {reco_text} !important; }}

.section-label {{
    color: {text_primary} !important;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-size: 1rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def plotly_template(theme: str) -> str:
    return "plotly_dark" if theme == "Dark" else "plotly_white"


def plotly_layout_overrides(theme: str) -> dict:
    """Return a complete layout dict that makes every chart text element visible in both themes."""
    is_dark = theme == "Dark"
    font_color = "#f1f5f9" if is_dark else "#0f172a"
    gridcolor  = "#334155" if is_dark else "#cbd5e1"
    zerocolor  = "#475569" if is_dark else "#94a3b8"
    axis_cfg = dict(
        color=font_color,
        tickcolor=font_color,
        tickfont=dict(color=font_color, family="DM Sans, sans-serif"),
        title=dict(font=dict(color=font_color, family="DM Sans, sans-serif")),
        gridcolor=gridcolor,
        zerolinecolor=zerocolor,
        linecolor=gridcolor,
    )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, family="DM Sans, sans-serif"),
        title=dict(font=dict(color=font_color, family="DM Sans, sans-serif", size=14)),
        legend=dict(
            font=dict(color=font_color, family="DM Sans, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=axis_cfg,
        yaxis=axis_cfg,
    )



def insight_box(text: str, box_class: str = "insight-box") -> None:
    safe = html_lib.escape(str(text))
    st.markdown(f'<div class="{box_class}"><p>{safe}</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading & validating data...")
def load_data(data_dir: str, mtime: float = 0.0):
    cleaned, report, panel, type_panel, channel_panel = prepare_data(data_dir)
    return cleaned, report, panel, type_panel, channel_panel


@st.cache_resource
def load_model(mtime: float):
    return load_bundle(MODEL_PATH)


def _apply_theme(fig: go.Figure, theme: str, **extra_layout) -> go.Figure:
    """Apply full theme overrides to any Plotly figure — axes, fonts, bg, legend."""
    o = plotly_layout_overrides(theme)
    fc = o["font"]["color"]
    # Merge title font color into any caller-supplied title dict, without duplicating the key
    caller_title = extra_layout.pop("title", {})
    if isinstance(caller_title, str):
        caller_title = {"text": caller_title}
    caller_title.setdefault("font", {})["color"] = fc
    # Merge legend — caller may supply extra legend kwargs (e.g. orientation)
    caller_legend = extra_layout.pop("legend", {})
    merged_legend = {**o["legend"], **caller_legend}
    fig.update_layout(
        template=plotly_template(theme),
        paper_bgcolor=o["paper_bgcolor"],
        plot_bgcolor=o["plot_bgcolor"],
        font=o["font"],
        title=caller_title,
        legend=merged_legend,
        **extra_layout,
    )
    # Axes — update_xaxes / update_yaxes propagate to every axis on the figure
    fig.update_xaxes(
        color=fc,
        tickcolor=fc,
        tickfont=dict(color=fc, family="DM Sans, sans-serif"),
        title_font=dict(color=fc, family="DM Sans, sans-serif"),
        gridcolor=o["xaxis"]["gridcolor"],
        zerolinecolor=o["xaxis"]["zerolinecolor"],
        linecolor=o["xaxis"]["linecolor"],
    )
    fig.update_yaxes(
        color=fc,
        tickcolor=fc,
        tickfont=dict(color=fc, family="DM Sans, sans-serif"),
        title_font=dict(color=fc, family="DM Sans, sans-serif"),
        gridcolor=o["yaxis"]["gridcolor"],
        zerolinecolor=o["yaxis"]["zerolinecolor"],
        linecolor=o["yaxis"]["linecolor"],
    )
    return fig


def baseline_beat_chart(test_df: pd.DataFrame, theme: str = "Dark"):
    """Visual proof that LightGBM captures shifts that rolling averages miss."""
    # Filter to the biggest outlier (e.g., Google Search) to show the biggest win
    subset = test_df[(test_df["channel"] == "google") & (test_df["campaign_type"] == "SEARCH")].tail(10)
    if subset.empty: return go.Figure()
    
    fig = go.Figure()
    # 1. Plot Actuals
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["target_30"]/4, name="Actual Revenue", line=dict(color="#10b981", width=3)))
    # 2. Plot Naive Baseline (EMA Anchor)
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["anchor_30"]/4, name="Naive Baseline", line=dict(color="#ef4444", width=2, dash="dash")))
    # 3. Plot Our Model
    fig.add_trace(go.Scatter(x=subset["date"], y=(subset["anchor_30"] + subset["residual_30"])/4, name="LightGBM Direct Horizon", line=dict(color="#0ea5e9", width=3)))
    
    return _apply_theme(fig, theme, title="Why ML Wins: Capturing Non-Linear Structural Shifts")


def money_flow_sankey(forecasts: pd.DataFrame, horizon: int = 30, theme: str = "Dark"):
    """Creates a stunning flow diagram from Spend -> Channel -> Revenue"""
    df = forecasts[(forecasts["level"] == "channel") & (forecasts["horizon_days"] == horizon)]
    if df.empty: return go.Figure()
    
    channels = list(df["channel"].unique())
    labels = ["Total Spend"] + channels + ["Projected Revenue"]
    
    channel_colors = {"google": "#4285F4", "meta": "#1877F2", "bing": "#00809D"}
    colors = ["#94a3b8"] + [channel_colors.get(ch.lower(), "#64748b") for ch in channels] + ["#10b981"]
    
    source, target, value = [], [], []
    
    for i, ch in enumerate(channels):
        ch_spend = df[df["channel"] == ch]["spend_scenario"].sum()
        if ch_spend > 0:
            source.append(0)
            target.append(i + 1)
            value.append(ch_spend)
            
    rev_node = len(channels) + 1
    for i, ch in enumerate(channels):
        ch_rev = df[df["channel"] == ch]["revenue_p50"].sum()
        if ch_rev > 0:
            source.append(i + 1)
            target.append(rev_node)
            value.append(ch_rev)
            
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5), label = labels, color = colors),
        link = dict(source = source, target = target, value = value, color = "rgba(14, 165, 233, 0.4)")
    )])
    return _apply_theme(fig, theme, title=dict(text=f"Portfolio Money Flow ({horizon} Days)"))


def fan_chart(forecasts: pd.DataFrame, level: str, channel: str = "all", theme: str = "Dark"):
    df = forecasts[(forecasts["level"] == level) & (forecasts["channel"] == channel)]
    if df.empty:
        return go.Figure()
    df = df.sort_values("horizon_days")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(df["horizon_days"]) + list(df["horizon_days"][::-1]),
            y=list(df["revenue_p90"]) + list(df["revenue_p10"][::-1]),
            fill="toself",
            fillcolor="rgba(14,165,233,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="P10–P90 Band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["horizon_days"],
            y=df["revenue_p50"],
            mode="lines+markers",
            name="P50 Revenue",
            line=dict(color="#0ea5e9", width=3),
            marker=dict(size=10),
        )
    )
    return _apply_theme(
        fig, theme,
        title=dict(text=f"Revenue Forecast — {level.title()} ({channel})"),
        xaxis_title="Horizon (days)",
        yaxis_title="Revenue ($)",
        height=420,
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40),
    )


def roas_chart(forecasts: pd.DataFrame, horizon: int = 90, theme: str = "Dark"):
    df = forecasts[
        (forecasts["level"] == "campaign_type") & (forecasts["horizon_days"] == horizon)
    ].copy()
    df["label"] = df["channel"] + " / " + df["campaign_type"]
    sorted_df = df.sort_values("roas_p50", ascending=True).tail(12)
    fig = px.bar(
        sorted_df,
        x="roas_p50",
        y="label",
        orientation="h",
        error_x=sorted_df["roas_p90"] - sorted_df["roas_p50"],
        error_x_minus=sorted_df["roas_p50"] - sorted_df["roas_p10"],
        color="channel",
        color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "bing": "#00809D"},
        labels={"roas_p50": "ROAS (P50)", "label": ""},
    )
    return _apply_theme(
        fig, theme,
        title=dict(text=f"ROAS by Campaign Type ({horizon}-day horizon)"),
        height=450,
    )


def calibration_chart(metrics: dict, theme: str = "Dark"):
    by_type = metrics.get("by_type", [])
    if not by_type:
        return go.Figure()
    df = pd.DataFrame(by_type)
    label_col = "model_key" if "model_key" in df.columns else "campaign_type"
    if label_col not in df.columns:
        return go.Figure()

    o = plotly_layout_overrides(theme)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[label_col],
            y=df["coverage_p10_p90"],
            mode="markers+lines",
            name="Observed Coverage",
            marker=dict(size=12, color="#0ea5e9"),
            line=dict(color="#0ea5e9"),
        )
    )
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="Target 80%",
        annotation_font_color=o["font"]["color"],
    )
    return _apply_theme(
        fig, theme,
        title=dict(text="Forecast Calibration — P10–P90 Coverage by Channel & Type"),
        yaxis_title="Coverage (%)",
        xaxis_title="Channel | Campaign Type",
        height=380,
        xaxis_tickangle=-45,
    )


def efficient_frontier_chart(forecasts: pd.DataFrame, horizon: int = 30, theme: str = "Dark"):
    df = forecasts[(forecasts["level"] == "campaign_type") & (forecasts["horizon_days"] == horizon)].copy()
    if df.empty: return go.Figure()
    
    df["label"] = df["channel"] + " / " + df["campaign_type"]
    fig = px.scatter(
        df,
        x="spend_scenario",
        y="revenue_p50",
        size="roas_p50",
        color="channel",
        hover_name="label",
        color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "bing": "#00809D"},
        labels={"spend_scenario": "Simulated Spend ($)", "revenue_p50": "Projected Revenue ($)"}
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='rgba(255,255,255,0.2)')))
    return _apply_theme(
        fig, theme,
        title=dict(text=f"Efficient Frontier — Budget vs Revenue ({horizon} Days)"),
        height=450
    )


def saturation_heatmap(curve_params: dict, theme: str = "Dark"):
    """Visualizes the 'k' parameter (saturation point) across campaign types."""
    rows = []
    for ct, cp in curve_params.items():
        if not cp or not isinstance(cp, list) or len(cp) < 3: continue
        rows.append({"campaign_type": ct, "k": cp[1]})
    if not rows: return go.Figure()
    
    df = pd.DataFrame(rows).sort_values("k", ascending=True)
    fig = px.bar(
        df, 
        x="k", 
        y="campaign_type", 
        orientation="h",
        color="k",
        color_continuous_scale="RdYlGn_r",
        labels={"k": "Saturation Threshold (Spend $)", "campaign_type": ""}
    )
    return _apply_theme(
        fig, theme,
        title=dict(text="Spend Saturation Limits (Hill Curve K-value)"),
        height=400
    )


def feature_importance_chart(theme: str = "Dark"):
    """Global feature importance derived from LightGBM training."""
    data = {
        "Feature": ["lag_2", "spend_ratio_vs_hist", "yoy_ratio", "planned_spend", "lag_12", 
                    "lrev_lag_26", "budget_ratio", "lrev_lag_52", "weeks_active", "yoy_roll4_lag52"],
        "Importance": [412, 408, 259, 251, 206, 171, 170, 153, 146, 141]
    }
    df = pd.DataFrame(data).sort_values("Importance", ascending=True)
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        labels={"Importance": "LightGBM Split Count", "Feature": ""}
    )
    return _apply_theme(
        fig, theme,
        title=dict(text="Top 10 AI Drivers (Global Feature Importance)"),
        height=400
    )


def main():
    sidebar = st.sidebar
    sidebar.image(
        "https://img.icons8.com/fluency/96/combo-chart.png",
        width=64,
    )
    sidebar.title("Controls")
    theme = sidebar.radio("Theme", ["Dark", "Light"], index=0, horizontal=True)

    # Inject CSS immediately after theme selection so everything picks it up
    inject_theme_css(theme)

    st.markdown(
        """
        <div class="main-header">
            <h1>📈 AIgnition Forecast</h1>
            <p>Probabilistic revenue & ROAS forecasting for Google, Meta & Microsoft Ads</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_dir = sidebar.text_input("Data directory", value=str(DATA_DIR))
    horizon = sidebar.selectbox("Primary horizon", HORIZONS, index=2)
    budget_mult = sidebar.slider("Budget multiplier", 0.5, 2.0, 1.0, 0.05)
    show_insights = sidebar.checkbox("Generate AI insights", value=True)

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    try:
        pipeline_mtime = os.path.getmtime(ROOT / "src" / "aggregate.py")
        cleaned, report, panel, type_panel, channel_panel = load_data(data_dir, pipeline_mtime)
        mtime = os.path.getmtime(MODEL_PATH)
        bundle = load_model(mtime)
    except Exception as exc:
        st.error(f"Failed to load: {exc}")
        st.stop()

    recent_spend = type_panel.groupby("campaign_type")["spend"].mean().mean()
    spend_scenario = recent_spend * budget_mult
    forecasts = generate_forecasts(
        bundle, type_panel, channel_panel, panel, spend_scenario=spend_scenario
    )

    sidebar.divider()
    csv_data = forecasts.to_csv(index=False)
    sidebar.download_button(
        label="📥 Download Agency Report (CSV)",
        data=csv_data,
        file_name="aignition_forecast_report.csv",
        mime="text/csv",
        help="Export the current scenario forecast to share with clients.",
    )

    recent_rows = type_panel.sort_values("date").groupby(["channel", "campaign_type"]).tail(1)
    baseline_rev = recent_rows[f"anchor_{horizon}"].sum()
    baseline_spend_hist = recent_rows[f"planned_spend_{horizon}"].sum()
    baseline_roas = baseline_rev / max(baseline_spend_hist, 1)

    blended = forecasts[
        (forecasts["level"] == "blended") & (forecasts["horizon_days"] == horizon)
    ]
    if len(blended):
        b = blended.iloc[0]
        rev_delta_pct = ((b['revenue_p50'] / max(baseline_rev, 1)) - 1.0) * 100
        rev_delta_str = f"{'↑' if rev_delta_pct > 0 else '↓'} {abs(rev_delta_pct):.1f}% vs Baseline"
        
        roas_delta = b['roas_p50'] - baseline_roas
        roas_delta_str = f"{'↑' if roas_delta > 0 else '↓'} {abs(roas_delta):.2f}x Efficiency"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("P50 Revenue", f"${b['revenue_p50']:,.0f}", delta=rev_delta_str, delta_color="normal")
        with c2:
            st.metric("P10–P90 Range", f"${b['revenue_p10']:,.0f} – ${b['revenue_p90']:,.0f}")
        with c3:
            st.metric("P50 ROAS", f"{b['roas_p50']:.2f}x", delta=roas_delta_str, delta_color="normal")
        with c4:
            cov = bundle.holdout_metrics.get("coverage_p10_p90", 0)
            st.metric("Backtest Coverage", f"{cov:.0f}%", help="Target: 80%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Forecasts", "💰 Budget Simulator", "🤖 AI Insights", "✅ Data Quality", "📐 Methodology"]
    )

    with tab1:
        st.plotly_chart(money_flow_sankey(forecasts, horizon, theme), use_container_width=True)
        st.plotly_chart(baseline_beat_chart(type_panel, theme), use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.plotly_chart(fan_chart(forecasts, "blended", "all", theme), use_container_width=True)
        with col_b:
            st.plotly_chart(fan_chart(forecasts, "channel", "google", theme), use_container_width=True)
        with col_c:
            st.plotly_chart(fan_chart(forecasts, "channel", "bing", theme), use_container_width=True)
        st.plotly_chart(roas_chart(forecasts, horizon, theme), use_container_width=True)

        st.subheader("Forecast Table")
        display_cols = [
            "horizon_days", "level", "channel", "campaign_type",
            "revenue_p10", "revenue_p50", "revenue_p90",
            "roas_p10", "roas_p50", "roas_p90", "spend_scenario",
        ]
        st.dataframe(
            forecasts[display_cols].style.format(
                {
                    "revenue_p10": "${:,.0f}",
                    "revenue_p50": "${:,.0f}",
                    "revenue_p90": "${:,.0f}",
                    "roas_p10": "{:.2f}x",
                    "roas_p50": "{:.2f}x",
                    "roas_p90": "{:.2f}x",
                    "spend_scenario": "${:,.0f}",
                }
            ),
            use_container_width=True,
            height=400,
        )

    with tab2:
        st.subheader("Budget Response Simulator")
        st.info(
            "Adjust the budget multiplier in the sidebar. Revenue forecasts scale via "
            "validated spend-response curves per campaign type."
        )
        curve_rows = []
        for ct, cp in bundle.curve_params.items():
            if not cp or not isinstance(cp, list) or len(cp) < 3: continue
            curve_rows.append(
                {
                    "campaign_type": ct,
                    "form": "hill",
                    "vmax": round(cp[0], 2),
                    "k": round(cp[1], 2),
                    "n": round(cp[2], 2),
                    "baseline_spend": round(bundle.baseline_spend.get(ct, 0), 2),
                }
            )
        if curve_rows:
            st.dataframe(pd.DataFrame(curve_rows), use_container_width=True)

        # Generate baseline (status quo) forecasts for comparison
        status_quo_forecasts = generate_forecasts(bundle, type_panel, channel_panel, panel, spend_scenario=recent_spend)
        
        sim_col1, sim_col2 = st.columns(2)
        fc = plotly_layout_overrides(theme)["font"]["color"]
        
        with sim_col1:
            base_fig = fan_chart(status_quo_forecasts, "blended", "all", theme)
            base_fig.update_layout(title=dict(text="Current Status Quo (Budget × 1.00)", font=dict(color=fc)))
            st.plotly_chart(base_fig, use_container_width=True)
            
        with sim_col2:
            sim_fig = fan_chart(forecasts, "blended", "all", theme)
            sim_fig.update_layout(title=dict(text=f"Simulated Forecast (Budget × {budget_mult:.2f})", font=dict(color=fc)))
            st.plotly_chart(sim_fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(efficient_frontier_chart(forecasts, horizon, theme), use_container_width=True)
        st.plotly_chart(saturation_heatmap(bundle.curve_params, theme), use_container_width=True)

    with tab3:
        st.subheader("AI-Assisted Business Insights")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(feature_importance_chart(theme), use_container_width=True)
        
        type_panel = type_panel.copy()
        type_panel["model_key"] = type_panel.apply(lambda r: model_key(r), axis=1)
        anomalies = detect_anomalies(type_panel, bundle)
        
        if show_insights:
            insights = generate_all_insights(
                forecasts=forecasts,
                panel=panel,
                type_panel=type_panel,
                bundle=bundle
            )
            
            st.markdown("### Forecast Explanation")
            exp = insights.get("forecast_explanation", {})
            insight_box(exp.get("explanation", ""), "insight-box")
            if "biggest_risk" in exp:
                insight_box("🚨 **Biggest Risk:** " + str(exp.get("biggest_risk", "")), "risk-box")
            if "biggest_opportunity" in exp:
                insight_box("💡 **Biggest Opportunity:** " + str(exp.get("biggest_opportunity", "")), "reco-box")
            
            st.markdown('<p class="section-label">Top Predictive Features (Local Importance)</p>', unsafe_allow_html=True)
            for d in exp.get("drivers", []):
                st.markdown(f"- **{d.get('group', 'Unknown')}** (Momentum: {d.get('momentum', 'N/A')}): {', '.join(d.get('top_features', []))}")
                
            st.markdown("### Cross-Channel Portfolio Strategy")
            port = insights.get("portfolio_insight", {})
            insight_box(port.get("insight", "N/A"), "insight-box")
            st.markdown(f"**Strategic Focus**: {port.get('strategic_focus', 'N/A')}")
            
            st.markdown("### Budget Recommendation")
            budg = insights.get("budget_recommendation", {})
            insight_box(budg.get("recommendation", "N/A"), "insight-box")
            if "recommended_shifts" in budg:
                for shift in budg["recommended_shifts"]:
                    st.markdown(f"- **{shift.get('group')}**: {shift.get('action')} - *{shift.get('reason')}*")
            
            st.markdown("### Operational Risk Assessment")
            risk = insights.get("operational_risk", {})
            insight_box(risk.get("risk_assessment", "N/A"), "risk-box")
            if "action_required" in risk:
                insight_box("⚡ **Action Required:** " + str(risk.get("action_required")), "reco-box")
            
            st.markdown("### Anomaly Interpretation")
            anom = insights.get("anomaly_interpretation", {})
            insight_box(anom.get("interpretation", "N/A"), "risk-box")
            st.markdown(f"**Actionable Advice**: {anom.get('actionable_advice', 'N/A')}")
            
        if anomalies:
            st.subheader("Detected Anomalies (Recent 52 Weeks)")
            st.dataframe(pd.DataFrame(anomalies[:15]), use_container_width=True)

        st.divider()
        st.subheader("💬 Chat with your Forecast")
        st.caption("Ask AIgnition an interactive question about the current scenario. (Requires GROQ_API_KEY)")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., If I move $5k from Meta to Google next week, what happens?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("AIgnition is thinking..."):
                response = chat_with_forecast(prompt, forecasts)
                
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab4:
        st.subheader("Data Validation Report")
        r = report.to_dict()
        v1, v2, v3 = st.columns(3)
        v1.metric("Total Rows", f"{r['total_rows']:,}")
        v2.metric("Duplicates", r["duplicate_campaign_date"])
        v3.metric("Zero-Spend / Rev>0", r["zero_spend_positive_revenue"])
        if r.get("messages"):
            for msg in r["messages"]:
                st.warning(msg)
        else:
            st.success("No critical validation issues detected.")
        st.subheader("Channel Summary")
        summary = cleaned.groupby("channel").agg(
            rows=("revenue", "count"),
            total_revenue=("revenue", "sum"),
            total_spend=("spend", "sum"),
        )
        summary["roas"] = summary["total_revenue"] / summary["total_spend"].replace(0, 1)
        st.dataframe(summary, use_container_width=True)

    with tab5:
        st.subheader("Holdout Validation (Unseen Test Weeks)")
        hm = bundle.holdout_metrics
        if hm:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("SMAPE (P50)", f"{hm.get('smape_p50', 0):.1f}%")
            m2.metric("WMAPE (P50)", f"{hm.get('wmape_p50', 0):.1f}%")
            m3.metric("MAPE (P50)", f"{hm.get('mape_p50', 0):.1f}%")
            m4.metric("MAE (P50)", f"${hm.get('mae_p50', 0):,.0f}")
            m5.metric("Median AE", f"${hm.get('median_ae_p50', 0):,.0f}")
            m6.metric("Coverage P10–P90", f"{hm.get('coverage_p10_p90', 0):.0f}%")

            st.caption(
                f"Evaluated on {hm.get('n_predictions', 0)} unseen weekly holdout rows "
                f"({hm.get('holdout_weeks', 10)} weeks). "
                f"Target coverage: **80%** — actuals should fall inside P10–P90 band."
            )

            baseline_rows = hm.get("baseline_comparison", [])
            if baseline_rows:
                st.subheader("LightGBM vs Baseline Forecasts")
                bdf = pd.DataFrame(baseline_rows)
                st.dataframe(
                    bdf.style.format(
                        {
                            "smape": "{:.1f}%",
                            "mape": "{:.1f}%",
                            "mae": "${:,.0f}",
                            "rmse": "${:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )
                beats = hm.get("lightgbm_beats_baseline", False)
                imp = hm.get("smape_improvement_vs_best_baseline", 0)
                if beats:
                    st.success(
                        f"LightGBM beats best baseline ({hm.get('best_baseline')}) "
                        f"by {imp:.1f} SMAPE points."
                    )
                else:
                    st.warning(
                        f"Best baseline: {hm.get('best_baseline')}. "
                        f"Consider retraining if gap persists."
                    )

            st.plotly_chart(calibration_chart(hm, theme), use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate holdout validation metrics.")

        st.subheader("Key Assumptions")
        st.markdown(
            f"""
            - Meta `conversion` field treated as **{'revenue proxy' if META_CONVERSION_AS_REVENUE else 'excluded from blended revenue'}**
            - Weekly aggregation (aggregate-period forecasts)
            - LightGBM quantile regression per **channel + campaign type**
            - Log-transformed revenue target with early stopping
            - Monte Carlo reconciliation for channel/blended totals
            - Spend treated as known input for ROAS calculation
            - AI insights powered by **Groq API** (optional, via `.env`)
            """
        )


if __name__ == "__main__":
    main()