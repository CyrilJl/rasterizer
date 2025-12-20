import math
import random
import time

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

from rasterizer import rasterize_lines, rasterize_polygons

GRID_SIZE = 1000
CELL_SIZE = 1.0
CRS = "EPSG:32631"
FIG_PATH = "benchmark.png"

SMALL_LINE_LENGTH = 15.0
BIG_LINE_LENGTH = 150.0
SMALL_POLY_RADIUS = 3.0
BIG_POLY_RADIUS = 30.0
MIN_ENTITIES = 10
MAX_ENTITIES = 5000
NUM_POINTS = 10
SEED = 0


def make_grid():
    x = np.arange(CELL_SIZE / 2.0, GRID_SIZE * CELL_SIZE, CELL_SIZE, dtype=float)
    y = np.arange(CELL_SIZE / 2.0, GRID_SIZE * CELL_SIZE, CELL_SIZE, dtype=float)
    return x, y, CRS


def log_spaced_counts(min_n, max_n, num_points):
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")
    values = np.logspace(math.log10(min_n), math.log10(max_n), num=num_points)
    values = np.unique(np.round(values).astype(int))
    return values[values > 0]


def generate_lines(n, x_range, y_range, length):
    lines = []
    half = length / 2.0
    for _ in range(n):
        cx = random.uniform(*x_range)
        cy = random.uniform(*y_range)
        angle = random.uniform(0.0, 2.0 * math.pi)
        dx = math.cos(angle) * half
        dy = math.sin(angle) * half
        x1 = min(max(cx - dx, x_range[0]), x_range[1])
        y1 = min(max(cy - dy, y_range[0]), y_range[1])
        x2 = min(max(cx + dx, x_range[0]), x_range[1])
        y2 = min(max(cy + dy, y_range[0]), y_range[1])
        lines.append(LineString([(x1, y1), (x2, y2)]))
    return lines


def generate_polygons(n, x_range, y_range, radius):
    polygons = []
    for _ in range(n):
        cx = random.uniform(*x_range)
        cy = random.uniform(*y_range)
        polygons.append(Point(cx, cy).buffer(radius))
    return polygons


def warmup(x, y, crs):
    line = LineString([(10.0, 10.0), (20.0, 20.0)])
    gdf_line = gpd.GeoDataFrame(geometry=[line], crs=crs)
    rasterize_lines(gdf_line, x, y, crs, mode="length")

    poly = Point(30.0, 30.0).buffer(2.0)
    gdf_poly = gpd.GeoDataFrame(geometry=[poly], crs=crs)
    rasterize_polygons(gdf_poly, x, y, crs, mode="area")


def time_call(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def benchmark_series(n_values, x, y, crs, line_length, poly_radius):
    x_range = (0.0, GRID_SIZE * CELL_SIZE)
    y_range = (0.0, GRID_SIZE * CELL_SIZE)
    line_times = []
    poly_times = []

    for n in n_values:
        lines = generate_lines(n, x_range, y_range, line_length)
        gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=crs)
        line_times.append(time_call(lambda: rasterize_lines(gdf_lines, x, y, crs, mode="length")))

        polygons = generate_polygons(n, x_range, y_range, poly_radius)
        gdf_polys = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        poly_times.append(time_call(lambda: rasterize_polygons(gdf_polys, x, y, crs, mode="area")))

    return line_times, poly_times


def print_table(label, n_values, line_times, poly_times):
    print(f"\nSize: {label}")
    print("n_entities, lines_s, polygons_s")
    for n, line_t, poly_t in zip(n_values, line_times, poly_times):
        print(f"{n}, {line_t:.6f}, {poly_t:.6f}")


def plot_results(n_values, small_line_times, small_poly_times, big_line_times, big_poly_times):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharex=True)

    axes[0].plot(n_values, small_line_times, marker="o", label="small")
    axes[0].plot(n_values, big_line_times, marker="o", label="big")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of lines (log scale)")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Lines")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(n_values, small_poly_times, marker="o", label="small")
    axes[1].plot(n_values, big_poly_times, marker="o", label="big")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of polygons (log scale)")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Polygons")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle("Rasterizer benchmark")
    fig.tight_layout()
    fig.savefig(FIG_PATH)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    x, y, crs = make_grid()

    warmup(x, y, crs)

    n_values = log_spaced_counts(MIN_ENTITIES, MAX_ENTITIES, NUM_POINTS)

    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} (cell size {CELL_SIZE})")
    print(f"Entity counts (log scale): {list(n_values)}")

    small_line_times, small_poly_times = benchmark_series(n_values, x, y, crs, SMALL_LINE_LENGTH, SMALL_POLY_RADIUS)
    print_table("small", n_values, small_line_times, small_poly_times)

    big_line_times, big_poly_times = benchmark_series(n_values, x, y, crs, BIG_LINE_LENGTH, BIG_POLY_RADIUS)
    print_table("big", n_values, big_line_times, big_poly_times)
    plot_results(n_values, small_line_times, small_poly_times, big_line_times, big_poly_times)
    print(f"\nSaved figure to: {FIG_PATH}")


if __name__ == "__main__":
    main()
