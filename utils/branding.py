"""Central brand palette and Plotly theming for the dashboard.

Premium dark theme: deep-navy gradient background, glassy panels, teal accents.
All charts pull their colours from here so the whole application keeps a
consistent, premium look. Tabs are styled large and high-contrast.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
# Brand palette (premium dark theme)
# --------------------------------------------------------------------------- #
NAVY = "#0A1A33"          # primary navy
NAVY_DEEP = "#060B18"     # darkest navy (page / heatmap floor)
NAVY_PANEL = "#0F2747"    # panel / card background
TEAL = "#14B8A6"          # primary brand teal
TEAL_BRIGHT = "#2DD4BF"   # accent teal (lines, highlights)
AMBER = "#F59E0B"         # REVIEW / warning
RED = "#EF4444"           # FREEZE / reject
GREEN = "#22C55E"         # CLEAR / pass
PURPLE = "#A78BFA"        # secondary accent
BLUE = "#60A5FA"          # tertiary accent
INK = "#E6EDF5"           # primary (light) text
MUTED = "#8FA3BF"         # muted text
GRID = "#26405E"          # inactive / dim elements + reference lines
GRIDLINE = "#16263C"      # faint chart gridlines

# Decision routing colours
DECISION_COLORS = {
    "CLEAR": GREEN,
    "REVIEW": AMBER,
    "FREEZE": RED,
}

# Sequential colour scale for heatmaps (dark navy -> teal -> bright).
SEQ_SCALE = [
    [0.0, NAVY_DEEP],
    [0.30, "#103E55"],
    [0.60, TEAL],
    [0.85, TEAL_BRIGHT],
    [1.0, "#CFFAF4"],
]

# Diverging scale for correlations (red -> panel -> teal).
DIVERGING_SCALE = [
    [0.0, RED],
    [0.5, NAVY_PANEL],
    [1.0, TEAL_BRIGHT],
]

# Qualitative palette for categorical series.
QUAL = [TEAL_BRIGHT, AMBER, PURPLE, BLUE, GREEN, RED, "#F472B6", INK]


def register_theme() -> str:
    """Register and return the name of the premium dark Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1A30",
        colorway=QUAL,
        title=dict(font=dict(color=INK, size=16)),
        xaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=MUTED)),
        legend=dict(font=dict(color=INK), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    pio.templates["aml_dark"] = template
    return "aml_dark"


THEME = register_theme()


def style(fig: go.Figure, height: int | None = None, title: str | None = None) -> go.Figure:
    """Apply the brand template to a figure and optionally set height/title."""
    fig.update_layout(template=THEME)
    if height is not None:
        fig.update_layout(height=height)
    if title is not None:
        fig.update_layout(title=title)
    return fig


# Streamlit page-wide CSS injected once from app.py
GLOBAL_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, #0E2A4A 0%, rgba(14,42,74,0) 55%),
            radial-gradient(1000px 500px at 110% 0%, #0C3B3B 0%, rgba(12,59,59,0) 50%),
            linear-gradient(160deg, #060B18 0%, #0A1A33 60%, #081427 100%);
        color: #E6EDF5;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1A33 0%, #081427 100%);
        border-right: 1px solid #16314F;
    }
    h1, h2, h3, h4 { color: #E6EDF5 !important; letter-spacing: 0.2px; }
    .brand-band {
        background: linear-gradient(100deg, #0A1A33 0%, #0E5C57 55%, #0A1A33 100%);
        border-radius: 16px; padding: 20px 26px; margin-bottom: 12px;
        border: 1px solid #16544E;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .brand-band h1 { margin: 0; font-size: 27px; color: #fff !important; }
    .brand-band p { margin: 5px 0 0 0; color: #B8E6E0; font-size: 14px; }

    /* ---- Premium, clearly-visible tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.035);
        padding: 8px; border-radius: 14px;
        border: 1px solid #16314F;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px; padding: 0 24px; border-radius: 11px;
        background: rgba(255,255,255,0.02);
        color: #9DB1CC; font-weight: 600; font-size: 15px;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(45,212,191,0.10); color: #CFFAF4;
        border-color: rgba(45,212,191,0.35);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #14B8A6 0%, #0E7490 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(20,184,166,0.35);
        border: 1px solid #2DD4BF !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

    /* Sub-section radio pills */
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.03); border: 1px solid #1C3454;
        padding: 6px 14px; border-radius: 10px; margin-right: 6px;
    }

    .pill {
        display: inline-block; padding: 4px 12px; margin: 3px;
        border-radius: 999px; font-size: 12px; font-weight: 600;
        background: rgba(45,212,191,0.14); color: #2DD4BF;
        border: 1px solid rgba(45,212,191,0.4);
    }
    .insight {
        background: rgba(255,255,255,0.035); border: 1px solid #1C3454;
        border-left: 4px solid #2DD4BF; border-radius: 10px;
        padding: 12px 16px; color: #C7D6EC; font-size: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }
    .insight b { color: #2DD4BF; }
    .badge-clear  { background:#0d2e1a; color:#22C55E; border:1px solid #22C55E;
                    padding:6px 14px; border-radius:8px; font-weight:700; }
    .badge-review { background:#3a2c08; color:#F59E0B; border:1px solid #F59E0B;
                    padding:6px 14px; border-radius:8px; font-weight:700; }
    .badge-freeze { background:#3a0d0d; color:#EF4444; border:1px solid #EF4444;
                    padding:6px 14px; border-radius:8px; font-weight:700; }
    .blink { animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.35; } }
    div[data-testid="stMetricValue"] { color:#2DD4BF; }
</style>
"""
