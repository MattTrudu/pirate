#!/usr/bin/env python3
"""Draw the presentation versions of grouper Figures 6 and 7 from saved timings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_COUNTS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
# Keep each batch size's colour and marker identical in the two figures.
STYLES = {
    1: ("#0072B2", "o"),
    2: ("#D55E00", "s"),
    10: ("#009E73", "^"),
    20: ("#56B4E9", "P"),
    30: ("#CC79A7", "D"),
    60: ("#4B4B4B", "v"),
}


def read_timings(path: Path) -> dict:
    """Validate complete, finite summaries before creating either output."""
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    rows = data.get("summaries")
    if not isinstance(rows, list):
        raise ValueError("The input must contain a 'summaries' list.")
    expected = {
        (figure, count, beams)
        for figure, beam_counts in ((6, (1, 2, 10, 30, 60)), (7, (60, 30, 20, 10, 2)))
        for count in CANDIDATE_COUNTS
        for beams in beam_counts
    }
    seen = set()
    for row in rows:
        key = (row["figure"], row["candidate_count"], row["beams_per_call"])
        if key not in expected:
            raise ValueError(f"Unexpected timing configuration: {key}")
        if key in seen:
            raise ValueError(f"Duplicate timing configuration: {key}")
        seen.add(key)
        values = [row[name] for name in ("p10_ms", "median_ms", "p90_ms")]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"Timings must be positive and finite: {key}")
        if values != sorted(values):
            raise ValueError(f"Expected p10 <= median <= p90: {key}")
        expected_calls = 1 if key[0] == 6 else 60 // key[2]
        if row.get("calls") != expected_calls:
            raise ValueError(f"Expected {expected_calls} calls for {key}")
    if seen != expected:
        raise ValueError(f"Missing timing configurations: {sorted(expected - seen)}")
    return data


def draw_figure(
    data: dict, figure_number: int, output_dir: Path, time_unit: str = "ms",
) -> list[Path]:
    """Save a single-axes plot, with a legend and a 10th–90th percentile band."""
    scale = 1.0
    if time_unit == "realtime-percent":
        budget_ms = float(data["metadata"]["chunk_budget_ms"])
        if not math.isfinite(budget_ms) or budget_ms <= 0:
            raise ValueError("chunk_budget_ms must be positive and finite.")
        scale = 100.0 / budget_ms
        unit_label = f" / {budget_ms / 1000:g} s (%)"
    elif time_unit == "ms":
        unit_label = " (ms)"
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")

    beam_counts = (1, 2, 10, 30, 60) if figure_number == 6 else (60, 30, 20, 10, 2)
    fig, ax = plt.subplots(figsize=(10.4, 6.2), layout="constrained")
    for beams in beam_counts:
        rows = sorted(
            (
                row for row in data["summaries"]
                if row["figure"] == figure_number and row["beams_per_call"] == beams
            ),
            key=lambda row: row["candidate_count"],
        )
        x = [row["candidate_count"] for row in rows]
        y = [row["median_ms"] * scale for row in rows]
        low = [row["p10_ms"] * scale for row in rows]
        high = [row["p90_ms"] * scale for row in rows]
        color, marker = STYLES[beams]
        label = (
            f"{beams} beam{'s' if beams != 1 else ''}"
            if figure_number == 6 else f"{60 // beams} × {beams} beams"
        )
        ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            x, y, label=label, color=color, marker=marker,
            linewidth=2.1, markersize=6, markeredgewidth=0.6,
            markeredgecolor="white",
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(CANDIDATE_COUNTS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(CANDIDATE_COUNTS[0] / 1.12, CANDIDATE_COUNTS[-1] * 1.12)
    ax.grid(which="major", axis="both", color="#D7DBDF", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#A3AAB0")
    ax.tick_params(axis="both", color="#A3AAB0", length=4, pad=6)

    if figure_number == 6:
        ax.set_title("Candidate concentration", loc="left", pad=17, weight="semibold")
        ax.set_xlabel("Total candidates per call", labelpad=10)
        ax.set_ylabel(f"Clustering time per call{unit_label}", labelpad=10)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.legend(title="Available beams", loc="upper left")
        stem = "figure6_clustering"
    else:
        ax.set_title("Batching 60 beams", loc="left", pad=17, weight="semibold")
        ax.set_xlabel("Candidates per beam", labelpad=10)
        ax.set_ylabel(f"Total clustering time{unit_label}", labelpad=10)
        ax.set_yscale("log")
        ticks = (
            (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
            if time_unit == "realtime-percent" else (3, 5, 10, 20, 50, 100, 200, 500)
        )
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.legend(title="Calls × beams per call", loc="upper left")
        stem = "figure7_batching"

    if time_unit == "realtime-percent":
        stem += "_realtime_percent"

    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Benchmark timings JSON")
    parser.add_argument(
        "--time-unit", choices=("ms", "realtime-percent"), default="ms",
        help="Y-axis units; percentages use metadata.chunk_budget_ms and separate filenames",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "docs",
        help="Directory for PNG and PDF figures (default: repository docs/)",
    )
    args = parser.parse_args()
    try:
        data = read_timings(args.input)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 19,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "none",
        "legend.labelspacing": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        for number in (6, 7):
            for path in draw_figure(data, number, args.output_dir, args.time_unit):
                print(path.resolve())


if __name__ == "__main__":
    main()
