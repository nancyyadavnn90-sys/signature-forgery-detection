"""
One consistent, colour-blind-safe chart style for the notebook, report and slides.

Colour roles (never swapped between charts):
    Genuine -> blue   #2a78d6
    Forged  -> orange #eb6834
Other categorical series follow a fixed, validated order; diverging maps use
blue <-> neutral grey <-> red; magnitudes use a single-hue blue ramp.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from . import config

GENUINE = "#2a78d6"
FORGED = "#eb6834"
CLASS_COLORS = {0: GENUINE, 1: FORGED, "Genuine": GENUINE, "Forged": FORGED}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b9b8b1"          # de-emphasised series (e.g. non-highlighted models)
# three-colour set for model-level charts (validated colour-blind safe, all pairs)
TRIO = ["#1baf7a", "#6250d6", "#eda100"]          # aqua, violet, yellow
FAMILY_COLORS = {"Baseline": "#eda100", "Single model": "#6250d6",
                 "Ensemble (bagging)": "#1baf7a", "Ensemble (boosting)": "#1baf7a",
                 "Ensemble (voting)": "#1baf7a"}
MAIN = "#6250d6"             # single-series model charts

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32f2f"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def save(fig, name: str):
    """Save a figure into outputs/figures (PNG, 200 dpi) and return the path."""
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, facecolor="white")
    return path


def class_legend(ax, loc="best", **kw):
    handles = [mpl.patches.Patch(color=GENUINE, label="Genuine"),
               mpl.patches.Patch(color=FORGED, label="Forged")]
    ax.legend(handles=handles, loc=loc, **kw)
