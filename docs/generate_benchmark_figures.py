import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from shapely import affinity, ops
from shapely.geometry import Point, Polygon, box

SAVE_PATH = Path("docs/_static")
CSV_PATH = SAVE_PATH / "polygon_threshold_benchmark.csv"
CURRENT_THRESHOLD = 81

OCCUPANCY_COLORS = {
    "0.00-0.15": "#b91c1c",
    "0.15-0.35": "#ea580c",
    "0.35-0.60": "#2563eb",
    "0.60-0.80": "#0891b2",
    "0.80-1.00": "#15803d",
}


def make_square():
    return box(-1.0, -1.0, 1.0, 1.0)


def make_disk():
    return Point(0.0, 0.0).buffer(1.0, quad_segs=48)


def make_diamond():
    return affinity.rotate(box(-1.0, -1.0, 1.0, 1.0), 45.0, origin=(0.0, 0.0))


def make_donut_light():
    return Point(0.0, 0.0).buffer(1.0, quad_segs=48).difference(Point(0.0, 0.0).buffer(0.45, quad_segs=48))


def make_donut_heavy():
    return Point(0.0, 0.0).buffer(1.0, quad_segs=48).difference(Point(0.0, 0.0).buffer(0.78, quad_segs=48))


def make_strip():
    return box(-1.3, -0.12, 1.3, 0.12)


def make_double_disk():
    left = affinity.translate(Point(0.0, 0.0).buffer(0.52, quad_segs=48), xoff=-0.78)
    right = affinity.translate(Point(0.0, 0.0).buffer(0.52, quad_segs=48), xoff=0.78)
    return ops.unary_union([left, right])


def make_frame():
    return box(-1.0, -1.0, 1.0, 1.0).difference(box(-0.725, -0.725, 0.725, 0.725))


def make_star():
    outer_radius = 1.0
    inner_radius = 0.38
    vertices = []
    for i in range(10):
        angle = math.radians(i * 36.0 - 90.0)
        radius = outer_radius if i % 2 == 0 else inner_radius
        vertices.append((radius * math.cos(angle), radius * math.sin(angle)))
    return Polygon(vertices)


def make_comb():
    base = box(-1.0, -0.275, 1.0, 0.275)
    teeth = []
    tooth_width = 0.26
    tooth_height = 0.72
    x_left = -1.0
    gap = (2.0 - 5 * tooth_width) / 4.0
    for i in range(5):
        xmin = x_left + i * (tooth_width + gap)
        xmax = xmin + tooth_width
        teeth.append(box(xmin, 0.275, xmax, 0.275 + tooth_height))
    return ops.unary_union([base, *teeth])


FAMILIES = [
    ("square", make_square(), "dense fill"),
    ("disk", make_disk(), "smooth boundary"),
    ("diamond", make_diamond(), "diagonal edges"),
    ("donut light", make_donut_light(), "one moderate hole"),
    ("donut heavy", make_donut_heavy(), "very low fill"),
    ("strip", make_strip(), "thin geometry"),
    ("double disk", make_double_disk(), "multipolygon"),
    ("frame", make_frame(), "ring-like shell"),
    ("star", make_star(), "concave outline"),
    ("comb", make_comb(), "many local boundary turns"),
]


def load_cases():
    cases = []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cases.append(
                {
                    "family": row["family"],
                    "mode": row["mode"],
                    "bbox_target": int(row["bbox_target"]),
                    "bbox_cells": int(row["bbox_cells"]),
                    "occupancy": float(row["occupancy"]),
                    "occupancy_bucket": row["occupancy_bucket"],
                    "exact_s": float(row["exact_s"]),
                    "hybrid_s": float(row["hybrid_s"]),
                }
            )
    return cases


def threshold_curve(cases):
    thresholds = sorted({0, *[case["bbox_cells"] for case in cases]})
    totals = []
    for threshold in thresholds:
        total = 0.0
        for case in cases:
            total += case["exact_s"] if case["bbox_cells"] <= threshold else case["hybrid_s"]
        totals.append(total)
    totals = np.asarray(totals)
    best_index = int(np.argmin(totals))
    best_threshold = thresholds[best_index]
    best_total = float(totals[best_index])
    current_total = 0.0
    for case in cases:
        current_total += case["exact_s"] if case["bbox_cells"] <= CURRENT_THRESHOLD else case["hybrid_s"]
    return thresholds, totals, best_threshold, best_total, current_total


def _plot_geometry(ax, geometry, facecolor):
    geoms = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    for geom in geoms:
        x, y = geom.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor="#0f172a", linewidth=1.5, alpha=0.9)
        for interior in geom.interiors:
            xi, yi = interior.xy
            ax.fill(xi, yi, facecolor="white", edgecolor="#0f172a", linewidth=1.2)


def generate_corpus_overview():
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 6.0))
    fig.patch.set_facecolor("#f8fafc")
    axes = axes.ravel()

    palette = [
        "#dbeafe",
        "#e0f2fe",
        "#fef3c7",
        "#fee2e2",
        "#fecaca",
        "#dcfce7",
        "#ede9fe",
        "#fae8ff",
        "#ffe4e6",
        "#e2e8f0",
    ]
    for ax, (name, geometry, note), color in zip(axes, FAMILIES, palette, strict=False):
        _plot_geometry(ax, geometry, color)
        ax.add_patch(plt.Rectangle((-1.02, -1.02), 2.04, 2.04, facecolor="none", edgecolor="#94a3b8", linewidth=1.0))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name.title(), fontsize=11, fontweight="bold", pad=6)
        ax.text(0.5, -0.12, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.8, color="#334155")

    fig.suptitle(
        "Benchmark Corpus: Occupancy, Holes, Concavity, And Boundary Complexity", fontsize=16, fontweight="bold"
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.11, wspace=0.18, hspace=0.5)
    fig.savefig(SAVE_PATH / "polygon_threshold_corpus.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_speedup_scatter(cases):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for bucket, color in OCCUPANCY_COLORS.items():
        matching = [case for case in cases if case["occupancy_bucket"] == bucket]
        for mode, marker in (("binary", "o"), ("area", "^")):
            mode_cases = [case for case in matching if case["mode"] == mode]
            if not mode_cases:
                continue
            ax.scatter(
                [case["bbox_cells"] for case in mode_cases],
                [case["exact_s"] / case["hybrid_s"] for case in mode_cases],
                s=36 if mode == "binary" else 44,
                marker=marker,
                facecolor=color,
                edgecolor="#0f172a",
                linewidth=0.4,
                alpha=0.78,
            )

    ax.axhline(1.0, color="#7c2d12", linewidth=1.8, linestyle=(0, (5, 4)))
    ax.set_xscale("log")
    ax.set_xlabel("Polygon bbox size on the target grid (cells)")
    ax.set_ylabel("Exact / Hybrid runtime")
    ax.set_title("Hybrid Speedup Across Bbox Size And Occupancy", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.6, alpha=0.6)

    occupancy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="#0f172a",
            markersize=7,
            label=bucket,
        )
        for bucket, color in OCCUPANCY_COLORS.items()
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#0f172a",
            markerfacecolor="white",
            linestyle="None",
            markersize=7,
            label="binary mode",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="#0f172a",
            markerfacecolor="white",
            linestyle="None",
            markersize=7,
            label="area mode",
        ),
    ]
    legend1 = ax.legend(handles=occupancy_handles, title="Occupancy bucket", loc="upper left", frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=mode_handles, loc="lower right", frameon=False)

    fig.savefig(SAVE_PATH / "polygon_threshold_speedup.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_threshold_curve(cases):
    thresholds, totals, best_threshold, best_total, current_total = threshold_curve(cases)
    thresholds_np = np.asarray(thresholds, dtype=float)
    relative_percent = (totals / best_total - 1.0) * 100.0
    current_percent = (current_total / best_total - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    ax.plot(thresholds_np, relative_percent, color="#2563eb", linewidth=2.2, marker="o", markersize=4)
    ax.axvline(best_threshold, color="#15803d", linewidth=1.8, linestyle=(0, (6, 4)))
    ax.axvline(CURRENT_THRESHOLD, color="#d97706", linewidth=1.8, linestyle=(0, (6, 4)))
    ax.scatter([best_threshold], [0.0], color="#15803d", s=54, zorder=5)
    ax.scatter([CURRENT_THRESHOLD], [current_percent], color="#d97706", s=54, zorder=5)

    ax.annotate(
        f"best observed threshold = {best_threshold}",
        xy=(best_threshold, 0.0),
        xytext=(best_threshold * 1.55 + 15, 0.055),
        arrowprops={"arrowstyle": "->", "color": "#15803d", "lw": 1.2},
        fontsize=9.5,
        color="#166534",
    )
    ax.annotate(
        f"current threshold = {CURRENT_THRESHOLD}\n(+{current_percent:.3f}% vs best)",
        xy=(CURRENT_THRESHOLD, current_percent),
        xytext=(CURRENT_THRESHOLD * 2.2 + 20, max(current_percent + 0.035, 0.06)),
        arrowprops={"arrowstyle": "->", "color": "#d97706", "lw": 1.2},
        fontsize=9.5,
        color="#92400e",
    )

    ax.set_xscale("symlog", linthresh=128)
    ax.set_xlabel("Decision threshold (bbox cells)")
    ax.set_ylabel("Aggregate runtime above best (%)")
    ax.set_title("Threshold Search: Aggregate Cost Is Flat Across Very Small Cutoffs", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.6, alpha=0.6)
    ax.set_ylim(bottom=-0.005)

    fig.savefig(SAVE_PATH / "polygon_threshold_curve.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    cases = load_cases()
    generate_corpus_overview()
    generate_speedup_scatter(cases)
    generate_threshold_curve(cases)


if __name__ == "__main__":
    main()
