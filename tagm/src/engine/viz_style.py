"""
Shared visualization constants and style utilities.
Okabe-Ito colorblind-safe palette, Tufte-inspired data-ink ratio,
dark theme per Material Design (#121212 base).
"""

import io
import base64
import textwrap
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# ─── Okabe-Ito Colorblind-Safe Category Palette ─────────────────
# Recommended by Nature Methods for categorical scientific data.
CAT_COLORS = {
    "benign":   "#0072B2",  # Blue
    "baseline": "#0072B2",
    "user_baseline": "#56B4E9",  # Sky blue (lighter variant)
    "mild":     "#E69F00",  # Amber
    "harmful":  "#D55E00",  # Vermillion
    "jailbreak":"#CC79A7",  # Reddish purple
    "adversarial": "#CC79A7",
    "unknown":  "#999999",
}

CAT_ORDER = ["benign", "mild", "harmful", "jailbreak"]

# Marker shapes for redundant encoding (accessibility)
CAT_MARKERS = {
    "benign": "o", "mild": "s", "harmful": "D", "jailbreak": "^",
    "baseline": "o", "unknown": "x",
}

# ─── Dark Theme Colors (Material Design #121212 base) ───────────
BG_DARK     = "#121212"   # Not pure black — prevents OLED bleed
BG_SURFACE  = "#1E1E1E"   # Elevated surface
BG_CARD     = "#252525"   # Card/panel background
TEXT_PRIMARY = "#DEE2E6"  # 87% white — less eye strain than pure white
TEXT_SECONDARY = "#9CA3AF" # 60% white
TEXT_MUTED   = "#6B7280"  # 38% white
GRID_COLOR   = "#333333"  # Subtle grid
SPINE_COLOR  = "#404040"  # Axis lines
ACCENT_CYAN  = "#56B4E9"  # Highlight/accent
ACCENT_GREEN = "#009E73"  # Positive/success

# ─── Continuous Colormaps ────────────────────────────────────────
# Sequential: Viridis (perceptually uniform, CVD-safe)
# Diverging: coolwarm centered at zero
TASM_CMAP = LinearSegmentedColormap.from_list(
    "tasm_seq", [BG_DARK, "#1a3a5c", "#2d6a8b", "#4a9fb5",
                 "#7db8c9", "#b8d4a0", "#e8c838", "#d55e00"])

# LTP-specific
SHAPE_COLORS = {"steep": "#E69F00", "flat": "#56B4E9", "inverted": "#D55E00"}

# ─── Effect Size Reference Lines ────────────────────────────────
EFFECT_SMALL  = 0.2
EFFECT_MEDIUM = 0.5
EFFECT_LARGE  = 0.8

# ─── Matplotlib RC Overrides for Publication Quality ────────────
TASM_RC = {
    "figure.facecolor": BG_DARK,
    "axes.facecolor": BG_SURFACE,
    "axes.edgecolor": SPINE_COLOR,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "text.color": TEXT_PRIMARY,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.facecolor": BG_DARK,
    "savefig.edgecolor": "none",
    "savefig.bbox": "tight",
    "legend.facecolor": BG_CARD,
    "legend.edgecolor": SPINE_COLOR,
    "legend.fontsize": 12,
    "legend.framealpha": 0.9,
}


def apply_style():
    """Apply TASM dark-theme RC params globally."""
    plt.rcParams.update(TASM_RC)


def fig_to_base64(fig) -> str:
    """Render figure to base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def placeholder_plot(message: str, title: str = "") -> str:
    """Render a centered message as a plot image.

    Used when a plot has nothing meaningful to show (e.g. a required category
    group is absent). Returning this informative empty-state instead of an
    empty string matters: an empty string surfaces to the user as a red
    "Failed to generate" error, which misreads as a crash rather than a
    "this view needs different data" condition.
    """
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    if title:
        ax.text(0.5, 0.74, title, ha="center", va="center", fontsize=15,
                fontweight="600", color=TEXT_PRIMARY, transform=ax.transAxes)
    ax.text(0.5, 0.42 if title else 0.5, message, ha="center", va="center",
            fontsize=12, color=TEXT_SECONDARY, wrap=True, transform=ax.transAxes)
    return fig_to_base64(fig)


def style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply clean Tufte-inspired styling to an axes object."""
    if title:
        ax.set_title(title, fontsize=16, fontweight="600", pad=10,
                     color=TEXT_PRIMARY)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, color=TEXT_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, color=TEXT_SECONDARY)


def cat_legend(ax, cats=None, loc="upper right", **kwargs):
    """Add a compact category legend."""
    if cats is None:
        cats = CAT_ORDER
    handles = [Patch(facecolor=CAT_COLORS.get(c, "#888"),
                     edgecolor="none", label=c.capitalize())
               for c in cats]
    ax.legend(handles=handles, loc=loc, fontsize=11, framealpha=0.85, **kwargs)


def wrap_label(text, width=20):
    return "\n".join(textwrap.wrap(str(text), width))


def wrap_labels(labels, width=20):
    return [wrap_label(l, width) for l in labels]


# Apply style on import
apply_style()
