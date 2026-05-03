import json
import os
import platform
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

from rasterizer import rasterize_polygons

matplotlib.use("Agg")

from matplotlib import pyplot as plt

DATASET_PATH = Path("dev/data/BDT_3-5_GPKG_LAMB93_D075-ED2026-03-15.gpkg")
LAYER_NAME = "batiment"
RESOLUTION_METERS = 10.0
IMAGE_PATH = Path("docs/_static/large_dataset_showcase.png")
METADATA_PATH = Path("docs/_static/large_dataset_showcase.json")


def load_buildings() -> tuple[gpd.GeoDataFrame, float]:
    start = time.perf_counter()
    gdf = gpd.read_file(DATASET_PATH, layer=LAYER_NAME, columns=[])
    return gdf, time.perf_counter() - start


def build_grid(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = bounds
    x = np.arange(xmin, xmax, RESOLUTION_METERS)
    y = np.arange(ymin, ymax, RESOLUTION_METERS)
    return x, y


def render_figure(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("Expected at least one positive raster cell in the showcase output.")

    vmin = float(np.percentile(positive, 33.0))
    vmax = float(np.percentile(positive, 100))
    norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin * 1.01))

    masked = np.ma.masked_less_equal(values, 0.0)
    extent = (
        float(x[0] - RESOLUTION_METERS / 2.0),
        float(x[-1] + RESOLUTION_METERS / 2.0),
        float(y[0] - RESOLUTION_METERS / 2.0),
        float(y[-1] + RESOLUTION_METERS / 2.0),
    )

    fig, ax = plt.subplots(figsize=(13.4, 9.2), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#e2e8f0")

    image = ax.imshow(masked, origin="lower", extent=extent, cmap="Spectral_r", norm=norm, interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Building footprint area per 10 m cell (m²)")

    ax.set_title("Paris Building Footprints Rasterized On A 10 m Grid", fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("Lambert-93 x (m)")
    ax.set_ylabel("Lambert-93 y (m)")
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(1.1)

    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "vmin_quantile_20": vmin,
        "vmax_quantile_99_8": vmax,
    }


def write_metadata(metadata: dict) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    total_start = time.perf_counter()
    gdf, read_seconds = load_buildings()
    x, y = build_grid(gdf.total_bounds)

    rasterize_start = time.perf_counter()
    raster = rasterize_polygons(polygons=gdf, x=x, y=y, crs=gdf.crs, mode="area", progress_bar=False)
    rasterize_seconds = time.perf_counter() - rasterize_start

    values = raster.values
    figure_stats = render_figure(values, x, y)
    total_seconds = time.perf_counter() - total_start

    metadata = {
        "dataset_path": str(DATASET_PATH).replace("\\", "/"),
        "layer_name": LAYER_NAME,
        "polygon_count": int(len(gdf)),
        "crs": str(gdf.crs),
        "resolution_m": RESOLUTION_METERS,
        "grid_shape": [int(len(y)), int(len(x))],
        "grid_cells": int(len(y) * len(x)),
        "bounds": {
            "xmin": float(gdf.total_bounds[0]),
            "ymin": float(gdf.total_bounds[1]),
            "xmax": float(gdf.total_bounds[2]),
            "ymax": float(gdf.total_bounds[3]),
        },
        "positive_cells": int(np.count_nonzero(values > 0)),
        "max_cell_area_m2": float(values.max()),
        "read_seconds": round(read_seconds, 3),
        "rasterize_seconds": round(rasterize_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "image_path": str(IMAGE_PATH).replace("\\", "/"),
        "color_scale": figure_stats,
    }
    write_metadata(metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
