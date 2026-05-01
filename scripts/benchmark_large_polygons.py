import importlib
import math
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rasterizer import rasterize_polygons

GRID_SIZE = 1000
CELL_SIZE = 1.0
CRS = "EPSG:32631"
SEED = 0
COUNTS = [1, 4, 16, 64]
SMALL_RADIUS = 4.0
LARGE_RADIUS = 120.0
EXACT_THRESHOLD = 10**9
HYBRID_THRESHOLD = 0


def make_grid():
    x = np.arange(CELL_SIZE / 2.0, GRID_SIZE * CELL_SIZE, CELL_SIZE, dtype=float)
    y = np.arange(CELL_SIZE / 2.0, GRID_SIZE * CELL_SIZE, CELL_SIZE, dtype=float)
    return x, y, CRS


def build_polygons(n, radius):
    polygons = []
    for _ in range(n):
        cx = random.uniform(radius + 5.0, GRID_SIZE * CELL_SIZE - radius - 5.0)
        cy = random.uniform(radius + 5.0, GRID_SIZE * CELL_SIZE - radius - 5.0)
        angle = random.uniform(0.0, 180.0)
        polygon = Point(cx, cy).buffer(radius, quad_segs=32)
        polygons.append(rotate(polygon, angle, origin="centroid"))
    return gpd.GeoDataFrame(geometry=polygons, crs=CRS)


def time_call(fn, repeats=3):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def benchmark_strategy(label, threshold, polygons_module, x, y, crs):
    polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = threshold
    print(f"\nStrategy: {label}")
    print("count,radius,time_s")

    for radius in (SMALL_RADIUS, LARGE_RADIUS):
        for count in COUNTS:
            gdf_polygons = build_polygons(count, radius)
            elapsed = time_call(lambda: rasterize_polygons(gdf_polygons, x, y, crs, mode="area"))
            print(f"{count},{radius:.1f},{elapsed:.6f}")


def warmup(polygons_module, x, y, crs):
    sample = build_polygons(1, SMALL_RADIUS)
    polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = EXACT_THRESHOLD
    rasterize_polygons(sample, x, y, crs, mode="area")
    polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = HYBRID_THRESHOLD
    rasterize_polygons(sample, x, y, crs, mode="area")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    polygons_module = importlib.import_module("rasterizer.polygons")
    x, y, crs = make_grid()
    warmup(polygons_module, x, y, crs)

    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} cells")
    benchmark_strategy("exact", EXACT_THRESHOLD, polygons_module, x, y, crs)
    benchmark_strategy("hybrid", HYBRID_THRESHOLD, polygons_module, x, y, crs)


if __name__ == "__main__":
    main()
