"""The charts that show what the analysis found.

Five figures, each answering one question:

- **the fundamental diagram** - does the measured speed and density behave like
  real traffic? This is the one that matters most. Speed against density on a
  real road traces a known curve, and if the numbers coming out of this pipeline
  trace it too, then they are measuring traffic rather than producing plausible
  noise. No amount of accuracy on the labels would show that.
- **the confusion matrix** - where the classifier's mistakes go. Mixing light up
  with medium is a borderline call; mixing light up with heavy means something is
  broken.
- **speed and density by level** - do the dataset's three words line up with the
  numbers, and how much do they overlap?
- **the day's profile** - when was the road busy? This is the part a traffic
  department would actually act on.
- **accuracy by weather** - does rain break it?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_flow.classify.evaluation import EvaluationReport
from traffic_flow.labels import CONGESTION_ORDER, CongestionLevel
from traffic_flow.metrics.speed import is_measured

# Imported from palette, not from matplotlib, so the file-rendering backend that
# palette selects is always in place before any figure is created.
from traffic_flow.viz.palette import (
    INK_SECONDARY,
    LEVEL_COLOURS,
    LEVEL_MARKERS,
    SEQUENTIAL_HUE,
    hide_spines,
    plt,
    use_report_style,
)

FIGURE_SIZE = (7.2, 4.6)


def _save(figure: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _level_groups(table: pd.DataFrame, column: str = "label"):
    """Yield each traffic level with its rows, always in light-medium-heavy order."""
    for level in CONGESTION_ORDER:
        rows = table[table[column].astype(str) == str(level)]
        if len(rows):
            yield level, rows


def fundamental_diagram(table: pd.DataFrame, out_dir: Path) -> Path:
    """Speed against density, one point per clip, marked by its true level.

    Traffic theory says these two are not independent: an empty road runs at free
    speed, and as vehicles pack in, speed falls away to a crawl. The curve that
    traces is the fundamental diagram, and it is the strongest evidence that the
    measurements mean something, because nothing in the pipeline was built to
    produce it.
    """
    use_report_style()
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)

    for level, rows in _level_groups(table):
        axes.scatter(
            rows["density_veh_per_km_per_lane"],
            rows["space_mean_speed_kph"],
            s=34,
            c=LEVEL_COLOURS[level],
            marker=LEVEL_MARKERS[level],
            linewidths=0.6,
            edgecolors="white",
            label=f"{level} ({len(rows)})",
            alpha=0.9,
        )

    axes.set_xlabel("density  (vehicles per km per lane)")
    axes.set_ylabel("space mean speed  (km/h)")
    axes.set_title("Speed falls as the road fills up")
    axes.legend(title="dataset label", loc="upper right")
    hide_spines(axes)
    return _save(figure, out_dir, "fundamental_diagram.png")


def confusion_matrix(report: EvaluationReport, out_dir: Path) -> Path:
    """Where the classifier's mistakes go, pooled over the four folds."""
    use_report_style()
    matrix = report.confusion()
    shares = matrix.to_numpy() / np.maximum(matrix.to_numpy().sum(axis=1, keepdims=True), 1)

    figure, axes = plt.subplots(figsize=(5.2, 4.4))
    axes.imshow(shares, cmap=SEQUENTIAL_HUE, vmin=0, vmax=1)
    axes.grid(False)

    labels = list(matrix.index)
    axes.set_xticks(range(len(labels)), labels)
    axes.set_yticks(range(len(labels)), labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")
    axes.set_title(f"{report.model_name}: {report.mean_accuracy:.1%} correct")

    # Every cell carries its number, so the shading never has to be decoded.
    for row in range(len(labels)):
        for column in range(len(labels)):
            count = matrix.iloc[row, column]
            axes.text(
                column,
                row,
                f"{count}\n{shares[row, column]:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if shares[row, column] > 0.55 else INK_SECONDARY,
            )
    hide_spines(axes, keep=())
    return _save(figure, out_dir, "confusion_matrix.png")


def levels_by_metric(table: pd.DataFrame, out_dir: Path) -> Path:
    """How speed and density are spread within each of the dataset's three words."""
    use_report_style()
    metrics = [
        ("space_mean_speed_kph", "space mean speed  (km/h)"),
        ("density_veh_per_km_per_lane", "density  (veh/km/lane)"),
    ]
    figure, axes_pair = plt.subplots(1, 2, figsize=(9.4, 4.2))

    for axes, (column, label) in zip(axes_pair, metrics, strict=True):
        groups = list(_level_groups(table))
        for position, (level, rows) in enumerate(groups):
            # Clips with no traffic have no speed. They are left off the chart
            # rather than drawn at zero, where they would look like a jam.
            values = rows[column].to_numpy(dtype=float)
            values = values[is_measured(values)]
            jitter = np.random.default_rng(0).normal(0, 0.06, size=len(values))
            axes.scatter(
                position + jitter,
                values,
                s=16,
                c=LEVEL_COLOURS[level],
                marker=LEVEL_MARKERS[level],
                alpha=0.55,
                linewidths=0,
            )
            median = float(np.median(values))
            axes.hlines(
                median, position - 0.28, position + 0.28,
                color=LEVEL_COLOURS[level], linewidth=2.5,
            )
            # Sat just clear of the line rather than on it, where the line itself
            # would cut through the digits.
            axes.annotate(
                f"{median:.0f}",
                xy=(position + 0.30, median),
                xytext=(4, 0),
                textcoords="offset points",
                va="center", ha="left", fontsize=9, fontweight="bold",
                color=LEVEL_COLOURS[level],
            )
        axes.set_xticks(range(len(groups)), [str(level) for level, _ in groups])
        axes.set_ylabel(label)
        hide_spines(axes)

    figure.suptitle("The dataset's three words, against what was measured", fontweight="bold")
    return _save(figure, out_dir, "levels_by_metric.png")


def day_profile(table: pd.DataFrame, out_dir: Path) -> Path:
    """Speed and flow through the two recorded days.

    Two stacked panels rather than two scales on one chart: speed in km/h and
    flow in vehicles per hour are different quantities, and drawing them against
    a shared axis would let the eye read a crossing point that does not exist.
    """
    use_report_style()
    hourly = (
        table.assign(hour=table["hour"])
        .groupby(["date", "hour"], as_index=False)
        .agg(
            speed=("space_mean_speed_kph", "median"),
            flow=("flow_veh_per_hour", "median"),
            clips=("clip_id", "size"),
        )
        .sort_values(["date", "hour"])
    )

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    dates = sorted(hourly["date"].unique())
    day_colours = [LEVEL_COLOURS[CongestionLevel.LIGHT], LEVEL_COLOURS[CongestionLevel.MEDIUM]]

    for date, colour in zip(dates, day_colours, strict=False):
        day = hourly[hourly["date"] == date]
        label = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        top.plot(day["hour"], day["speed"], color=colour, linewidth=2, marker="o",
                 markersize=4, label=label)
        bottom.plot(day["hour"], day["flow"], color=colour, linewidth=2, marker="o",
                    markersize=4, label=label)

    top.set_ylabel("median speed  (km/h)")
    top.set_title("Speed and flow through the two recorded days")
    bottom.set_ylabel("flow  (vehicles per hour)")
    bottom.set_xlabel("hour of day")
    top.legend(title="date", loc="lower left")
    for axes in (top, bottom):
        hide_spines(axes)
    return _save(figure, out_dir, "day_profile.png")


def accuracy_by_weather(
    report: EvaluationReport, baseline: EvaluationReport, out_dir: Path
) -> Path:
    """Does the weather break it - measured against what guessing would score.

    The model's accuracy on its own is misleading here, and the chart would be
    dishonest without the second bar. Nearly every rain clip in this dataset is
    light traffic, so answering "light" every time is already right 95% of the
    time when it rains. A model scoring 96% in the rain has therefore barely
    beaten a constant, while the same model scoring 92% in clear weather - where
    guessing gets 47% - is doing almost all of the work.
    """
    use_report_style()
    model = report.accuracy_by("weather")
    guess = baseline.accuracy_by("weather")
    order = model.index

    figure, axes = plt.subplots(figsize=(7.0, 3.8))
    positions = np.arange(len(order))
    height = 0.36

    axes.barh(positions + height / 2, model.loc[order, "accuracy"], height=height,
              color=LEVEL_COLOURS[CongestionLevel.LIGHT], label=report.model_name)
    axes.barh(positions - height / 2, guess.loc[order, "accuracy"], height=height,
              color=LEVEL_COLOURS[CongestionLevel.MEDIUM], label="always answer 'light'")

    axes.set_yticks(positions, [f"{w}\n(n={int(model.loc[w, 'n_clips'])})" for w in order])
    axes.set_xlim(0, 1.12)
    axes.set_xlabel("accuracy")
    axes.set_title("Clear weather is where the model earns its keep")
    axes.grid(axis="y", visible=False)
    axes.legend(loc="lower right")

    for position, weather in zip(positions, order, strict=True):
        for offset, table in ((height / 2, model), (-height / 2, guess)):
            axes.text(table.loc[weather, "accuracy"] + 0.012, position + offset,
                      f"{table.loc[weather, 'accuracy']:.0%}",
                      va="center", fontsize=9, color=INK_SECONDARY)
    hide_spines(axes)
    return _save(figure, out_dir, "accuracy_by_weather.png")


def speed_density_flow(table: pd.DataFrame, out_dir: Path) -> Path:
    """Flow against density - the other half of the fundamental diagram.

    Flow rises with density up to the road's capacity, then falls again as the
    traffic jams: the same number of vehicles per hour can mean a quiet road or a
    failing one. That turning point is what a traffic department watches for.
    """
    use_report_style()
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)

    for level, rows in _level_groups(table):
        flow = rows["density_veh_per_km_per_lane"] * rows["space_mean_speed_kph"]
        axes.scatter(
            rows["density_veh_per_km_per_lane"], flow,
            s=34, c=LEVEL_COLOURS[level], marker=LEVEL_MARKERS[level],
            linewidths=0.6, edgecolors="white", alpha=0.9,
            label=f"{level} ({len(rows)})",
        )

    axes.set_xlabel("density  (vehicles per km per lane)")
    axes.set_ylabel("flow  (vehicles per hour per lane)")
    axes.set_title("Flow peaks at the road's capacity, then collapses")
    axes.legend(title="dataset label", loc="upper right")
    hide_spines(axes)
    return _save(figure, out_dir, "flow_vs_density.png")


def build_all(
    table: pd.DataFrame,
    report: EvaluationReport,
    baseline: EvaluationReport,
    out_dir: str | Path,
) -> list[Path]:
    """Write every figure and return where they went."""
    out_dir = Path(out_dir)
    figures = [
        fundamental_diagram(table, out_dir),
        speed_density_flow(table, out_dir),
        levels_by_metric(table, out_dir),
        confusion_matrix(report, out_dir),
        accuracy_by_weather(report, baseline, out_dir),
    ]
    if "hour" in table and "date" in table:
        figures.append(day_profile(table, out_dir))
    return figures
