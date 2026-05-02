import argparse
import csv
import importlib
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import affinity, ops
from shapely.geometry import Point, Polygon, box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rasterizer import rasterize_polygons

GRID_SIZE = 256
CELL_SIZE = 1.0
CELL_AREA = CELL_SIZE * CELL_SIZE
CRS = "EPSG:32631"
EXACT_THRESHOLD = 10**9
HYBRID_THRESHOLD = 0
REPEATS = 7
JITTERS = (
    (-0.31, 0.27, 0.0),
    (0.23, -0.19, 7.0),
    (0.11, 0.34, 13.0),
)
BBOX_TARGETS = (36, 64, 100, 144, 196, 256, 400, 576, 784, 1024, 1600, 2500, 4096, 6400, 9216, 12544, 16384)
MODES = ("binary", "area")


@dataclass(frozen=True)
class BenchmarkCase:
    family: str
    mode: str
    bbox_target: int
    bbox_cells: int
    occupancy: float
    exact_s: float
    hybrid_s: float


def make_grid() -> tuple[np.ndarray, np.ndarray, str]:
    x = np.arange(CELL_SIZE / 2.0, GRID_SIZE * CELL_SIZE, CELL_SIZE, dtype=np.float64)
    y = np.arange(CELL_SIZE / 2.0, GRID_SIZE * CELL_SIZE, CELL_SIZE, dtype=np.float64)
    return x, y, CRS


def _centered_box(width: float, height: float):
    return box(-width / 2.0, -height / 2.0, width / 2.0, height / 2.0)


def make_square():
    return _centered_box(2.0, 2.0)


def make_disk():
    return Point(0.0, 0.0).buffer(1.0, quad_segs=16)


def make_diamond():
    return affinity.rotate(_centered_box(2.0, 2.0), 45.0, origin=(0.0, 0.0))


def make_donut_light():
    return Point(0.0, 0.0).buffer(1.0, quad_segs=16).difference(Point(0.0, 0.0).buffer(0.45, quad_segs=16))


def make_donut_heavy():
    return Point(0.0, 0.0).buffer(1.0, quad_segs=16).difference(Point(0.0, 0.0).buffer(0.78, quad_segs=16))


def make_strip():
    return _centered_box(2.6, 0.24)


def make_double_disk():
    left = affinity.translate(Point(0.0, 0.0).buffer(0.52, quad_segs=16), xoff=-0.78)
    right = affinity.translate(Point(0.0, 0.0).buffer(0.52, quad_segs=16), xoff=0.78)
    return ops.unary_union([left, right])


def make_frame():
    return _centered_box(2.0, 2.0).difference(_centered_box(1.45, 1.45))


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
    base = _centered_box(2.0, 0.55)
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


FAMILIES = {
    "square": make_square,
    "disk": make_disk,
    "diamond": make_diamond,
    "donut_light": make_donut_light,
    "donut_heavy": make_donut_heavy,
    "strip": make_strip,
    "double_disk": make_double_disk,
    "frame": make_frame,
    "star": make_star,
    "comb": make_comb,
}


def scale_to_bbox_area(geometry, bbox_target: int):
    target_side = math.sqrt(bbox_target)
    minx, miny, maxx, maxy = geometry.bounds
    width = maxx - minx
    height = maxy - miny
    return affinity.scale(geometry, xfact=target_side / width, yfact=target_side / height, origin=(0.0, 0.0))


def position_geometry(geometry, dx: float, dy: float, rotation: float):
    geometry = affinity.rotate(geometry, rotation, origin=(0.0, 0.0))
    center = GRID_SIZE * CELL_SIZE / 2.0
    return affinity.translate(geometry, xoff=center + dx, yoff=center + dy)


def time_call(fn, repeats: int = REPEATS) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def compute_bbox_cells(geometry, x: np.ndarray, y: np.ndarray) -> int:
    xmin, ymin, xmax, ymax = geometry.bounds
    half_dx = CELL_SIZE / 2.0
    half_dy = CELL_SIZE / 2.0
    ix_start = np.searchsorted(x, xmin - half_dx, side="right") - 1
    ix_end = np.searchsorted(x, xmax + half_dx, side="left") + 1
    iy_start = np.searchsorted(y, ymin - half_dy, side="right") - 1
    iy_end = np.searchsorted(y, ymax + half_dy, side="left") + 1
    ix_start = max(0, ix_start)
    iy_start = max(0, iy_start)
    ix_end = min(len(x), ix_end)
    iy_end = min(len(y), iy_end)
    return (ix_end - ix_start) * (iy_end - iy_start)


def occupancy_bucket(occupancy: float) -> str:
    if occupancy < 0.15:
        return "0.00-0.15"
    if occupancy < 0.35:
        return "0.15-0.35"
    if occupancy < 0.60:
        return "0.35-0.60"
    if occupancy < 0.80:
        return "0.60-0.80"
    return "0.80-1.00"


def warmup(polygons_module, x: np.ndarray, y: np.ndarray, crs: str) -> None:
    geometry = position_geometry(scale_to_bbox_area(make_disk(), 256), 0.0, 0.0, 0.0)
    sample = gpd.GeoDataFrame(geometry=[geometry], crs=crs)
    for mode in MODES:
        polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = EXACT_THRESHOLD
        rasterize_polygons(sample, x, y, crs, mode=mode)
        polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = HYBRID_THRESHOLD
        rasterize_polygons(sample, x, y, crs, mode=mode)


def build_case_geometry(family: str, bbox_target: int, jitter: tuple[float, float, float]):
    dx, dy, rotation = jitter
    geometry = scale_to_bbox_area(FAMILIES[family](), bbox_target)
    return position_geometry(geometry, dx, dy, rotation)


def benchmark_case(
    polygons_module,
    family: str,
    bbox_target: int,
    mode: str,
    jitter: tuple[float, float, float],
    x: np.ndarray,
    y: np.ndarray,
    crs: str,
):
    geometry = build_case_geometry(family, bbox_target, jitter)
    bbox_cells = compute_bbox_cells(geometry, x, y)
    occupancy = geometry.area / (bbox_cells * CELL_AREA)
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=crs)

    polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = EXACT_THRESHOLD
    exact_s = time_call(lambda: rasterize_polygons(gdf, x, y, crs, mode=mode))

    polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = HYBRID_THRESHOLD
    hybrid_s = time_call(lambda: rasterize_polygons(gdf, x, y, crs, mode=mode))

    polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = EXACT_THRESHOLD
    exact = rasterize_polygons(gdf, x, y, crs, mode=mode)
    polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS = HYBRID_THRESHOLD
    hybrid = rasterize_polygons(gdf, x, y, crs, mode=mode)
    np.testing.assert_allclose(hybrid.values, exact.values)

    return BenchmarkCase(
        family=family,
        mode=mode,
        bbox_target=bbox_target,
        bbox_cells=bbox_cells,
        occupancy=occupancy,
        exact_s=exact_s,
        hybrid_s=hybrid_s,
    )


def choose_threshold_by_total(cases: list[BenchmarkCase]) -> tuple[int, float]:
    candidate_thresholds = sorted({0, *[case.bbox_cells for case in cases]})
    best_threshold = candidate_thresholds[0]
    best_total = math.inf
    for threshold in candidate_thresholds:
        total = 0.0
        for case in cases:
            total += case.exact_s if case.bbox_cells <= threshold else case.hybrid_s
        if total < best_total:
            best_total = total
            best_threshold = threshold
    return best_threshold, best_total


def choose_threshold_by_ratio(cases: list[BenchmarkCase]) -> tuple[int, float]:
    candidate_thresholds = sorted({0, *[case.bbox_cells for case in cases]})
    best_threshold = candidate_thresholds[0]
    best_ratio = math.inf
    for threshold in candidate_thresholds:
        ratios = []
        for case in cases:
            selected = case.exact_s if case.bbox_cells <= threshold else case.hybrid_s
            oracle = min(case.exact_s, case.hybrid_s)
            ratios.append(selected / oracle)
        mean_ratio = statistics.fmean(ratios)
        if mean_ratio < best_ratio:
            best_ratio = mean_ratio
            best_threshold = threshold
    return best_threshold, best_ratio


def summarize(cases: list[BenchmarkCase], current_threshold: int) -> None:
    print("family,mode,bbox_target,bbox_cells,occupancy,occupancy_bucket,exact_ms,hybrid_ms,hybrid_speedup")
    for case in cases:
        speedup = case.exact_s / case.hybrid_s
        print(
            f"{case.family},{case.mode},{case.bbox_target},{case.bbox_cells},{case.occupancy:.3f},"
            f"{occupancy_bucket(case.occupancy)},{case.exact_s * 1000:.3f},{case.hybrid_s * 1000:.3f},{speedup:.3f}"
        )

    best_total_threshold, best_total = choose_threshold_by_total(cases)
    best_ratio_threshold, best_ratio = choose_threshold_by_ratio(cases)
    current_total = sum(case.exact_s if case.bbox_cells <= current_threshold else case.hybrid_s for case in cases)
    current_ratio = statistics.fmean(
        [
            (case.exact_s if case.bbox_cells <= current_threshold else case.hybrid_s) / min(case.exact_s, case.hybrid_s)
            for case in cases
        ]
    )

    print()
    print(f"Current threshold: {current_threshold} cells -> aggregate median time {current_total * 1000:.3f} ms")
    print(f"Best threshold by aggregate time: {best_total_threshold} cells -> {best_total * 1000:.3f} ms")
    print(f"Current threshold: {current_threshold} cells -> mean slowdown vs oracle {current_ratio:.4f}x")
    print(f"Best threshold by oracle ratio: {best_ratio_threshold} cells -> {best_ratio:.4f}x")

    print()
    print("Occupancy summary")
    print("bucket,cases,exact_ms,hybrid_ms,hybrid_wins")
    buckets = sorted({occupancy_bucket(case.occupancy) for case in cases})
    for bucket in buckets:
        matching = [case for case in cases if occupancy_bucket(case.occupancy) == bucket]
        exact_ms = sum(case.exact_s for case in matching) * 1000.0
        hybrid_ms = sum(case.hybrid_s for case in matching) * 1000.0
        hybrid_wins = sum(case.hybrid_s < case.exact_s for case in matching)
        print(f"{bucket},{len(matching)},{exact_ms:.3f},{hybrid_ms:.3f},{hybrid_wins}")

    print()
    print("Per-size summary")
    print("bbox_cells,exact_ms,hybrid_ms,preferred")
    seen = sorted({case.bbox_cells for case in cases})
    for bbox_cells in seen:
        matching = [case for case in cases if case.bbox_cells == bbox_cells]
        exact_ms = sum(case.exact_s for case in matching) * 1000.0
        hybrid_ms = sum(case.hybrid_s for case in matching) * 1000.0
        preferred = "hybrid" if hybrid_ms < exact_ms else "exact"
        print(f"{bbox_cells},{exact_ms:.3f},{hybrid_ms:.3f},{preferred}")


def write_cases_csv(cases: list[BenchmarkCase], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "family",
                "mode",
                "bbox_target",
                "bbox_cells",
                "occupancy",
                "occupancy_bucket",
                "exact_s",
                "hybrid_s",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.family,
                    case.mode,
                    case.bbox_target,
                    case.bbox_cells,
                    f"{case.occupancy:.12f}",
                    occupancy_bucket(case.occupancy),
                    f"{case.exact_s:.12f}",
                    f"{case.hybrid_s:.12f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    polygons_module = importlib.import_module("rasterizer.polygons")
    current_threshold = polygons_module._HYBRID_POLYGON_THRESHOLD_CELLS
    x, y, crs = make_grid()
    warmup(polygons_module, x, y, crs)

    cases = []
    for family in FAMILIES:
        for mode in MODES:
            for bbox_target in BBOX_TARGETS:
                for jitter in JITTERS:
                    cases.append(benchmark_case(polygons_module, family, bbox_target, mode, jitter, x, y, crs))

    if args.csv_output is not None:
        write_cases_csv(cases, args.csv_output)

    summarize(cases, current_threshold)


if __name__ == "__main__":
    main()
