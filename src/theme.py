"""Look and feel of the dashboard: light and dark themes, CSS, and small inline graphics.

One set of role names (page, card, ink, accent ...) with a light and a dark value
each.  The stylesheet is written once against those roles; switching mode swaps
the values, Streamlit's own widget theme and the chart palette together.
"""

from __future__ import annotations

import numpy as np

# Streamlit's widget theme per mode (applied at run time, see app.apply_theme).
STREAMLIT_THEME = {
    "light": {"base": "light", "primaryColor": "#2a78d6", "backgroundColor": "#f5f7fa",
              "secondaryBackgroundColor": "#eaeef4", "textColor": "#0b0b0b"},
    "dark": {"base": "dark", "primaryColor": "#3987e5", "backgroundColor": "#0d0f13",
             "secondaryBackgroundColor": "#222733", "textColor": "#f3f3ef"},
}

ROLES = {
    "light": {
        "page": "#f5f7fa", "card": "#ffffff", "sidebar": "#ffffff", "soft": "#eef2f7",
        "border": "rgba(15,23,42,0.09)", "shadow": "0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.06)",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "accent": "#2a78d6", "accent-soft": "#e8f1fc", "accent-ink": "#1c5cab",
        "up": "#006300", "down": "#b42318",
        "plan-bg": "#fdf0ea", "plan-border": "#f0b79f", "plan-ink": "#8a3a16",
        "hero": "linear-gradient(120deg, #0b2545 0%, #134a8e 55%, #2a78d6 100%)",
    },
    "dark": {
        "page": "#0d0f13", "card": "#171a20", "sidebar": "#12151a", "soft": "#1f232c",
        "border": "rgba(255,255,255,0.09)", "shadow": "0 1px 2px rgba(0,0,0,0.5), 0 10px 28px rgba(0,0,0,0.35)",
        "ink": "#f5f5f2", "ink2": "#c3c2b7", "muted": "#8f8d84",
        "accent": "#3987e5", "accent-soft": "rgba(57,135,229,0.16)", "accent-ink": "#9ec5f4",
        "up": "#4cc779", "down": "#f08a8a",
        "plan-bg": "rgba(217,89,38,0.20)", "plan-border": "rgba(233,121,63,0.55)", "plan-ink": "#f6c1a6",
        "hero": "linear-gradient(120deg, #081a33 0%, #0f3a73 55%, #1f62bb 100%)",
    },
}

_CSS = """
<style>
:root {__VARS__}
.stApp {background: var(--page);}
.block-container {padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stSidebar"] {background: var(--sidebar); border-right: 1px solid var(--border);}

/* hero */
.hero {background: var(--hero); border-radius: 18px; padding: 22px 26px 18px 26px; margin-bottom: 16px;
    position: relative; overflow: hidden; box-shadow: var(--shadow);}
.hero::after {content: ""; position: absolute; right: -80px; top: -120px; width: 380px; height: 380px;
    border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,0.20) 0%, rgba(255,255,255,0) 65%);}
.hero .row {display: flex; align-items: center; gap: 14px; position: relative; z-index: 1;}
.hero .mark {width: 44px; height: 44px; border-radius: 12px; background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.30); display: flex; align-items: center; justify-content: center;}
.hero .title {font-size: 1.55rem; font-weight: 660; color: #ffffff; line-height: 1.15; letter-spacing: -0.01em;}
.hero .sub {color: rgba(255,255,255,0.80); font-size: 0.93rem; margin-top: 2px;}
.chips {margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; position: relative; z-index: 1;}
.chip {display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem;
    color: rgba(255,255,255,0.86); border: 1px solid rgba(255,255,255,0.24); background: rgba(255,255,255,0.12);}
.chip b {color: #ffffff; font-weight: 620;}
.chip.plan {background: #eb6834; border-color: #f39a73; color: #ffffff;}

/* stat tiles */
.tile {background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px 12px 16px;
    min-height: 108px; box-shadow: var(--shadow); transition: transform .15s ease, box-shadow .15s ease;}
.tile:hover {transform: translateY(-2px);}
.tile .t-head {display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;}
.tile .label {color: var(--ink2); font-size: 0.84rem;}
.tile .ico {width: 28px; height: 28px; border-radius: 8px; background: var(--accent-soft); color: var(--accent);
    display: flex; align-items: center; justify-content: center; flex: none;}
.tile .value {color: var(--ink); font-size: 1.55rem; font-weight: 640; line-height: 1.25; white-space: nowrap;
    letter-spacing: -0.01em;}
.tile .sub {color: var(--ink2); font-size: 0.82rem; margin-top: 3px;}
.tile .up {color: var(--up); font-weight: 600;} .tile .down {color: var(--down); font-weight: 600;}
.tile .value.long {font-size: 1.22rem; line-height: 1.6;}
.tile .viz {margin-top: auto; padding-top: 8px;}
.tile.kpi {min-height: 186px; display: flex; flex-direction: column;}
.tile.mini {min-height: 0; padding: 10px 14px; box-shadow: none;}
.tile.mini .value {font-size: 1.15rem;}
.tile.on {border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); background: var(--accent-soft);}
.rng {position: relative; height: 8px; border-radius: 999px; background: var(--accent-soft);}
.rng .fill {position: absolute; top: 0; height: 8px; border-radius: 999px; background: var(--accent); opacity: .45;}
.rng .dot {position: absolute; top: -3px; width: 14px; height: 14px; margin-left: -7px; border-radius: 50%;
    background: var(--accent); border: 2px solid var(--card);}
.rng-lab {display: flex; justify-content: space-between; color: var(--muted); font-size: 0.7rem; margin-top: 4px;}

/* cards: every bordered container created by section() / card() */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[class*="st-key-card_"]) {
    background: var(--card); border: 1px solid var(--border); border-radius: 16px; box-shadow: var(--shadow);
    padding: 18px 18px 14px 18px;}
.sec-title {font-size: 1.03rem; font-weight: 640; color: var(--ink); margin-bottom: 2px; letter-spacing: -0.005em;}
.sec-cap {color: var(--ink2); font-size: 0.85rem; margin-bottom: 6px;}
.keymsg {display: flex; gap: 12px; align-items: flex-start; background: var(--card); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 14px; padding: 14px 18px; margin: 0 0 14px 0;
    box-shadow: var(--shadow);}
.keymsg .ico {width: 32px; height: 32px; border-radius: 9px; background: var(--accent-soft); color: var(--accent);
    display: flex; align-items: center; justify-content: center; flex: none;}
.keymsg .k {color: var(--accent-ink); font-size: 0.72rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase;}
.keymsg .h {font-size: 1.12rem; font-weight: 620; color: var(--ink); line-height: 1.35; margin-top: 1px;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
    border: 1px solid var(--border); color: var(--ink2); background: var(--soft); margin-right: 6px;}
.headline {font-size: 1.22rem; font-weight: 640; line-height: 1.35; color: var(--ink); margin: 8px 0 6px 0;}
.small {color: var(--ink2); font-size: 0.86rem;}
.big {font-size: 2.2rem; font-weight: 660; color: var(--ink); line-height: 1.1; letter-spacing: -0.02em;}
.overline {color: var(--muted); font-size: 0.72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    margin: 10px 0 2px 0;}
.brand {display: flex; align-items: center; gap: 10px; margin-bottom: 2px;}
.brand .mark {width: 34px; height: 34px; border-radius: 10px; background: var(--hero); display: flex;
    align-items: center; justify-content: center;}
.brand .name {font-size: 1.12rem; font-weight: 680; color: var(--ink); letter-spacing: -0.01em;}
.brand .tag {color: var(--ink2); font-size: 0.8rem;}
.foot {color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 26px;}

/* tabs as a segmented control */
div[data-baseweb="tab-list"] {background: var(--soft); padding: 5px; border-radius: 14px; gap: 4px;
    border: 1px solid var(--border);}
button[data-baseweb="tab"] {border-radius: 10px; padding: 6px 16px; height: auto; color: var(--ink2);}
button[data-baseweb="tab"] p {font-size: 0.95rem; font-weight: 560;}
button[data-baseweb="tab"][aria-selected="true"] {background: var(--card); color: var(--ink);
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none;}

/* buttons and inputs */
.stButton > button, .stDownloadButton > button {border-radius: 10px; border: 1px solid var(--border); font-weight: 550;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color: var(--accent); color: var(--accent);}
.stButton > button[kind="primary"] {background: var(--accent); border-color: var(--accent); color: #ffffff;}
.stButton > button[kind="primary"]:hover {filter: brightness(1.08); color: #ffffff;}
[data-testid="stExpander"] details {border-radius: 14px; border-color: var(--border); background: var(--card);}
</style>
"""

LOGO = ("<svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='#ffffff' stroke-width='2' "
        "stroke-linecap='round' stroke-linejoin='round'><path d='M3 17l5-5 4 3 8-9'/><path d='M15 6h5v5'/>"
        "<path d='M3 21h18' opacity='.55'/></svg>")

_ICON_PATHS = {
    "trend": "<path d='M2 11.5l3.5-3.5 3 2.5L14 4.5'/><path d='M10.5 4.5H14V8'/>",
    "range": "<path d='M2 8h12'/><path d='M4.5 5.5L2 8l2.5 2.5'/><path d='M11.5 5.5L14 8l-2.5 2.5'/>",
    "spend": "<circle cx='8' cy='8' r='6'/><path d='M8 4.6v6.8'/><path d='M9.8 6.3c-.4-.5-1-.8-1.8-.8-1 0-1.8.5-1.8 "
             "1.2 0 1.7 3.7.9 3.7 2.6 0 .7-.8 1.2-1.9 1.2-.8 0-1.5-.3-1.9-.9'/>",
    "ratio": "<path d='M3.5 12.5l9-9'/><circle cx='4.6' cy='4.6' r='1.7'/><circle cx='11.4' cy='11.4' r='1.7'/>",
    "bulb": "<path d='M5.5 9.8a4 4 0 1 1 5 0c-.5.4-.8 1-.8 1.6H6.3c0-.6-.3-1.2-.8-1.6z'/><path d='M6.5 13.6h3'/>",
}


def icon(name: str, size: int = 16) -> str:
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 16 16' fill='none' stroke='currentColor' "
            f"stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>{_ICON_PATHS[name]}</svg>")


def css(dark: bool) -> str:
    """The stylesheet with the role values of the chosen mode."""
    roles = ROLES["dark" if dark else "light"]
    variables = " ".join(f"--{name}: {value};" for name, value in roles.items())
    return _CSS.replace("__VARS__", variables)


def sparkline(values, dark: bool, height: int = 30) -> str:
    """Tiny trend line for a stat tile: the context series in grey, its last point in the accent."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return ""
    roles = ROLES["dark" if dark else "light"]
    width = 120.0
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    x = np.linspace(3, width - 3, len(v))
    y = height - 4 - (v - lo) / span * (height - 8)
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    area = f"M{x[0]:.1f},{height} L" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + f" L{x[-1]:.1f},{height} Z"
    return (f"<svg width='100%' height='{height}' viewBox='0 0 {width:.0f} {height}' preserveAspectRatio='none'>"
            f"<path d='{area}' fill='{roles['accent']}' opacity='0.10'/>"
            f"<polyline points='{line}' fill='none' stroke='{roles['muted']}' stroke-width='1.6' "
            f"vector-effect='non-scaling-stroke'/>"
            f"<line x1='{x[-1]:.1f}' y1='{y[-1]:.1f}' x2='{x[-1]:.1f}' y2='{y[-1]:.1f}' stroke='{roles['accent']}' "
            f"stroke-width='6' stroke-linecap='round' vector-effect='non-scaling-stroke'/></svg>")


def range_bar(lo: float, mid: float, hi: float) -> str:
    """P10-P90 as a filled span on a track, with the P50 as a dot."""
    top = max(hi * 1.06, 1e-9)
    a, m, b = (float(np.clip(v / top * 100, 0, 100)) for v in (lo, mid, hi))
    return (f"<div class='rng'><div class='fill' style='left:{a:.1f}%;width:{max(b - a, 1):.1f}%'></div>"
            f"<div class='dot' style='left:{m:.1f}%'></div></div>"
            "<div class='rng-lab'><span>P10</span><span>P50</span><span>P90</span></div>")
