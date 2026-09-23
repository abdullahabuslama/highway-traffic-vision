"""Colours and styling for the report charts.

The three traffic levels are drawn in blue, orange and aqua rather than the
green-amber-red a traffic light would use. Green and red are the obvious choice
and they are the wrong one: to a reader with the commonest form of colour
blindness those two are nearly the same colour, and the whole chart collapses.
The trio here was checked with a colour-vision simulator and the closest pair
still reads clearly apart.

Green, amber and red do appear on the annotated video, where the banner spells
out the word "HEAVY" next to the colour. There the colour decorates a label that
already says the answer, so nothing depends on telling the hues apart.

Charts here are rendered for a light page. The marks are thin, the grid is faint,
and every series is labelled as well as coloured, so identity never rests on
colour alone.
"""

from __future__ import annotations

import matplotlib

# Render straight to files, never to a window. Without this, matplotlib looks for
# a desktop toolkit and crashes on a server, in a container, or in CI - anywhere
# the report is most likely to be generated. It must be set before pyplot is
# imported, which is why it sits above that import rather than with the rest.
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import rcParams

from traffic_flow.labels import CongestionLevel

#: One colour per traffic level. Checked in a colour-vision simulator: the
#: closest pair of these three is comfortably distinguishable.
LEVEL_COLOURS: dict[CongestionLevel, str] = {
    CongestionLevel.LIGHT: "#2a78d6",
    CongestionLevel.MEDIUM: "#eb6834",
    CongestionLevel.HEAVY: "#1baf7a",
}

#: A different marker shape per level as well, so the chart still works in
#: greyscale, in print, and for a reader who sees no colour at all.
LEVEL_MARKERS: dict[CongestionLevel, str] = {
    CongestionLevel.LIGHT: "o",
    CongestionLevel.MEDIUM: "s",
    CongestionLevel.HEAVY: "^",
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

#: Single hue, light to dark, for anything that shows a magnitude - the
#: confusion matrix, mostly. One hue, never a rainbow.
SEQUENTIAL_HUE = "Blues"


def use_report_style() -> None:
    """Set the look of every chart in one place, so they read as one set."""
    rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "figure.dpi": 130,
        }
    )


def hide_spines(axes: plt.Axes, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    """Drop the box around a chart, keeping only the axes that carry meaning."""
    for side, spine in axes.spines.items():
        spine.set_visible(side in keep)
