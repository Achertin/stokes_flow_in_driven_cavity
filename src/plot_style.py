"""Shared projector-friendly plot style for the presentation figures.

Usage:
    from src.plot_style import apply_style, annotate_slopes, save
    apply_style()               # call once, before creating any figure
"""

import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIR = ("/home/lukasl/Documents/vienna_uni/computational_analysis_seminar/"
             "comp_math_presentation/images")

# Font size for slope / rate labels written next to plot segments
ANNOTATION_SIZE = 20


def apply_style():
    """Large, bold text and thick lines, readable on a projector."""
    plt.rcParams.update({
        # LaTeX text; \boldmath makes all math bold (tick labels, symbols)
        "text.usetex": True,
        "text.latex.preamble": r"\boldmath",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        # sizes
        "font.size": 28,
        "axes.titlesize": 30,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        # lines and markers
        "lines.linewidth": 3,
        "lines.markersize": 12,
        # frame and ticks
        "axes.linewidth": 2.5,
        "xtick.major.width": 2.5, "ytick.major.width": 2.5,
        "xtick.major.size": 8,    "ytick.major.size": 8,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
        "xtick.minor.size": 4,    "ytick.minor.size": 4,
        # grid and legend
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.linestyle": "--",
        "grid.linewidth": 1.2,
        "grid.alpha": 0.6,
        "legend.framealpha": 0.9,
    })


def annotate_slopes(ax, x, y, color, offset=(0, 12), fmt="{:.2f}"):
    """Write the log-log slope between successive points at each segment's midpoint."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    slopes = np.log(y[1:] / y[:-1]) / np.log(x[1:] / x[:-1])
    for i, s in enumerate(slopes):
        xm, ym = np.sqrt(x[i] * x[i + 1]), np.sqrt(y[i] * y[i + 1])
        ax.annotate(r"$\mathbf{" + fmt.format(s) + "}$", (xm, ym),
                    textcoords="offset points", xytext=offset,
                    ha="center", fontsize=ANNOTATION_SIZE, color=color)
    return slopes


def save(fig, filename):
    """Save into the presentation's image folder."""
    fig.savefig(f"{IMAGE_DIR}/{filename}", dpi=300, bbox_inches="tight", transparent=True)
