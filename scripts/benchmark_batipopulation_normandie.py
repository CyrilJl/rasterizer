import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rasterizer import rasterize_polygons

DATASET_PATH = Path("dev/data/BatiPopulation_2021_Normandie/BatiPopulation_2021_Normandie.TAB")
WEIGHT_COLUMN = "Popt2021"
RESOLUTION_METERS = 15.0
CRS = "EPSG:2154"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark weighted rasterization of BatiPopulation_2021_Normandie on a 15 m grid."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--resolution", type=float, default=RESOLUTION_METERS)
    parser.add_argument("--weight-column", default=WEIGHT_COLUMN)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None, help="Optional smoke-test limit before full runs.")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def build_axis(min_coord: float, max_coord: float, resolution: float) -> np.ndarray:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    count = int(math.ceil((max_coord - min_coord) / resolution))
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    return min_coord + resolution / 2.0 + np.arange(count, dtype=np.float64) * resolution


def build_grid(bounds: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = bounds
    return build_axis(xmin, xmax, resolution), build_axis(ymin, ymax, resolution)


def warmup_numba(weight_column: str, resolution: float) -> float:
    x = build_axis(0.0, 240.0, resolution)
    y = build_axis(0.0, 240.0, resolution)
    polygons = gpd.GeoDataFrame(
        {weight_column: [2.0, 8.0]},
        geometry=[
            box(20.0, 20.0, 40.0, 40.0),
            box(60.0, 60.0, 210.0, 210.0),
        ],
        crs=CRS,
    )

    start = time.perf_counter()
    rasterize_polygons(polygons, x=x, y=y, crs=CRS, mode="area", weight=weight_column)
    return time.perf_counter() - start


def load_dataset(path: Path, weight_column: str, max_features: int | None) -> tuple[gpd.GeoDataFrame, float]:
    start = time.perf_counter()
    read_kwargs = {"columns": [weight_column]}
    if max_features is not None:
        read_kwargs["rows"] = max_features
    gdf = gpd.read_file(path, **read_kwargs)
    return gdf, time.perf_counter() - start


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    positive = values > 0.0
    return {
        "positive_cells": int(np.count_nonzero(positive)),
        "max_value": float(values.max(initial=0.0)),
        "sum_value": float(values.sum()),
    }


def benchmark_once(
    gdf: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    weight_column: str,
    progress_bar: bool,
) -> dict[str, float | int]:
    start = time.perf_counter()
    raster = rasterize_polygons(
        polygons=gdf,
        x=x,
        y=y,
        crs=gdf.crs,
        mode="area",
        weight=weight_column,
        progress_bar=progress_bar,
    )
    rasterize_seconds = time.perf_counter() - start

    stats_start = time.perf_counter()
    stats = summarize_values(raster.values)
    stats_seconds = time.perf_counter() - stats_start

    del raster
    gc.collect()

    return {
        "rasterize_seconds": rasterize_seconds,
        "stats_seconds": stats_seconds,
        **stats,
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")

    warmup_seconds = warmup_numba(args.weight_column, args.resolution)
    gdf, read_seconds = load_dataset(args.dataset, args.weight_column, args.max_features)
    x, y = build_grid(gdf.total_bounds, args.resolution)

    runs = []
    for repeat in range(args.repeats):
        result = benchmark_once(gdf, x, y, args.weight_column, args.progress_bar)
        result["repeat"] = repeat + 1
        runs.append(result)
        print(
            "repeat={repeat} rasterize_seconds={rasterize_seconds:.3f} "
            "positive_cells={positive_cells} sum_value={sum_value:.6f}".format(**result),
            flush=True,
        )

    raster_times = [run["rasterize_seconds"] for run in runs]
    metadata = {
        "dataset_path": str(args.dataset).replace("\\", "/"),
        "feature_count": int(len(gdf)),
        "crs": str(gdf.crs),
        "weight_column": args.weight_column,
        "resolution_m": args.resolution,
        "grid_shape": [int(len(y)), int(len(x))],
        "grid_cells": int(len(y) * len(x)),
        "raster_bytes_float64": int(len(y) * len(x) * np.dtype(np.float64).itemsize),
        "bounds": {
            "xmin": float(gdf.total_bounds[0]),
            "ymin": float(gdf.total_bounds[1]),
            "xmax": float(gdf.total_bounds[2]),
            "ymax": float(gdf.total_bounds[3]),
        },
        "numba_warmup_seconds_excluded": warmup_seconds,
        "read_seconds": read_seconds,
        "runs": runs,
        "best_rasterize_seconds": min(raster_times),
        "median_rasterize_seconds": float(np.median(raster_times)),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }

    text = json.dumps(metadata, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
