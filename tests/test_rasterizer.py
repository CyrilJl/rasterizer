import importlib
import random
import time

import geopandas as gpd
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from rasterizer import rasterize_lines, rasterize_polygons

np.random.seed(0)

# Common setup for tests
CRS = "EPSG:32631"  # UTM 31N, metric CRS
X_RANGE = (0, 100)
Y_RANGE = (0, 100)
DX = 1.0
DY = 1.0
X = np.arange(X_RANGE[0] + DX / 2, X_RANGE[1], DX)
Y = np.arange(Y_RANGE[0] + DY / 2, Y_RANGE[1], DY)
TIMING_GRID_SIZE = 2000
TIMING_POLYGON_COUNT = 300_000
TIMING_REPEATS = 5


@pytest.fixture
def grid():
    """Create a grid for the tests."""
    return {"x": X, "y": Y, "crs": CRS}


@pytest.fixture
def grid_gdf(grid):
    """Create a GeoDataFrame for the grid cells."""
    cells = []
    for i, y in enumerate(grid["y"]):
        for j, x in enumerate(grid["x"]):
            cells.append(
                {
                    "geometry": box(x - DX / 2, y - DY / 2, x + DX / 2, y + DY / 2),
                    "row": i,
                    "col": j,
                }
            )
    return gpd.GeoDataFrame(cells, crs=grid["crs"])


def generate_random_lines(n_geometries, x_range, y_range):
    """Generate a list of random LineString and MultiLineString geometries."""
    geometries = []
    for _ in range(n_geometries):
        num_points = random.randint(2, 10)
        points = []
        for _ in range(num_points):
            points.append((random.uniform(*x_range), random.uniform(*y_range)))

        if random.random() < 0.2:  # 20% chance of being a MultiLineString
            num_lines = random.randint(2, 5)
            lines = []
            for _ in range(num_lines):
                num_points_in_line = random.randint(2, 10)
                line_points = []
                for _ in range(num_points_in_line):
                    line_points.append((random.uniform(*x_range), random.uniform(*y_range)))
                lines.append(LineString(line_points))
            geometries.append(MultiLineString(lines))
        else:
            geometries.append(LineString(points))

    return geometries


def generate_random_polygons(n_geometries, x_range, y_range, with_interiors_fraction=0.3):
    """Generate a list of random Polygon and MultiPolygon geometries."""
    geometries = []
    for i in range(n_geometries):
        num_points = random.randint(5, 15)
        points = np.random.rand(num_points, 2)
        points[:, 0] = points[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        points[:, 1] = points[:, 1] * (y_range[1] - y_range[0]) + y_range[0]

        try:
            hull = ConvexHull(points)
            exterior = points[hull.vertices]
            poly = Polygon(exterior)
        except Exception:
            continue

        if i < n_geometries * with_interiors_fraction:
            interior_points = poly.centroid.coords[0] + (points - poly.centroid.coords[0]) * 0.5
            try:
                interior_hull = ConvexHull(interior_points)
                interior = interior_points[interior_hull.vertices]
                if Polygon(interior).is_valid:
                    poly = Polygon(exterior, [interior])
            except Exception:
                pass

        if random.random() < 0.2:  # 20% chance of being a MultiPolygon
            num_polys = random.randint(2, 5)
            polys = []
            for _ in range(num_polys):
                num_points_in_poly = random.randint(5, 15)
                poly_points = np.random.rand(num_points_in_poly, 2)
                poly_points[:, 0] = poly_points[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
                poly_points[:, 1] = poly_points[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
                try:
                    hull = ConvexHull(poly_points)
                    polys.append(Polygon(poly_points[hull.vertices]))
                except Exception:
                    continue
            geometries.append(MultiPolygon(polys))
        else:
            geometries.append(poly)

    return geometries


def generate_timing_polygons(n_geometries, grid_size, rng):
    """Generate a mix of rectangles, triangles, and concave polygons for the timing benchmark."""
    widths = rng.uniform(1.0, 50.0, size=n_geometries)
    heights = rng.uniform(1.0, 50.0, size=n_geometries)
    half_widths = widths / 2.0
    half_heights = heights / 2.0
    centers_x = rng.uniform(half_widths, grid_size - half_widths)
    centers_y = rng.uniform(half_heights, grid_size - half_heights)
    shape_kinds = rng.integers(0, 3, size=n_geometries)

    geometries = []
    for kind, cx, cy, half_w, half_h in zip(shape_kinds, centers_x, centers_y, half_widths, half_heights):
        if kind == 0:
            geometries.append(box(cx - half_w, cy - half_h, cx + half_w, cy + half_h))
        elif kind == 1:
            geometries.append(
                Polygon(
                    [
                        (cx, cy + half_h),
                        (cx - half_w, cy - half_h),
                        (cx + half_w, cy - half_h),
                    ]
                )
            )
        else:
            geometries.append(
                Polygon(
                    [
                        (cx - half_w, cy - half_h),
                        (cx + half_w, cy - half_h),
                        (cx + half_w, cy + half_h),
                        (cx, cy + half_h / 3.0),
                        (cx - half_w, cy + half_h),
                    ]
                )
            )
    return gpd.GeoSeries(geometries, crs=CRS)


def test_rasterize_lines(grid, grid_gdf):
    """
    Test the correctness of line rasterization by comparing with geopandas overlay.
    """
    # Generate random lines
    lines = generate_random_lines(50, X_RANGE, Y_RANGE)
    gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=CRS)

    # Use geopandas overlay to get the expected lengths
    overlay = gpd.overlay(grid_gdf, gdf_lines.explode(), how="intersection", keep_geom_type=False)
    overlay["length"] = overlay.geometry.length
    expected_lengths = overlay.groupby(["row", "col"])["length"].sum().reset_index()
    expected_lengths = expected_lengths.merge(grid_gdf[["row", "col"]], on=["row", "col"], how="right")
    expected_lengths = expected_lengths.fillna(0)["length"].values.reshape((len(Y), len(X)))

    # Rasterize with mode='length'
    raster_len = rasterize_lines(gdf_lines, **grid, mode="length")
    np.testing.assert_allclose(raster_len.values, expected_lengths, atol=1e-3)

    # Rasterize with mode='binary' and check for consistency
    raster_bin = rasterize_lines(gdf_lines, **grid, mode="binary")
    expected_bin = expected_lengths > 0
    np.testing.assert_array_equal(raster_bin.values, expected_bin)


def test_rasterize_lines_geoseries_infers_crs(grid):
    gdf_lines = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 10)]), LineString([(10, 0), (10, 10)])],
        crs=CRS,
    )

    raster_gdf = rasterize_lines(gdf_lines, x=grid["x"], y=grid["y"], mode="length")
    raster_series = rasterize_lines(gdf_lines.geometry, x=grid["x"], y=grid["y"], mode="length")

    np.testing.assert_allclose(raster_series.values, raster_gdf.values)
    assert str(raster_series.rio.crs) == CRS


def test_rasterize_lines_without_any_crs_is_supported(grid):
    geometry = [LineString([(0, 0), (10, 10)]), LineString([(10, 0), (10, 10)])]
    gdf_lines = gpd.GeoDataFrame(geometry=geometry, crs=None)
    expected = rasterize_lines(gpd.GeoDataFrame(geometry=geometry, crs=CRS), **grid, mode="length")

    raster = rasterize_lines(gdf_lines, x=grid["x"], y=grid["y"], mode="length")

    np.testing.assert_allclose(raster.values, expected.values)
    assert raster.rio.crs is None


def test_rasterize_lines_rejects_non_uniform_grid(grid):
    gdf_lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 10)])], crs=CRS)
    bad_x = np.array([0.5, 1.5, 3.0], dtype=float)

    with pytest.raises(ValueError, match="x must be evenly spaced."):
        rasterize_lines(gdf_lines, x=bad_x, y=grid["y"], mode="length")


def test_rasterize_grid_aligned_lines_preserve_boundary_semantics(grid):
    gdf_lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(1, 0), (1, 3)]),
            LineString([(0, 1), (3, 1)]),
        ],
        crs=CRS,
    )

    raster_len = rasterize_lines(gdf_lines, **grid, mode="length")
    assert raster_len.values[0, 0] == 0.0
    assert raster_len.values[0, 1] == 1.0
    assert raster_len.values[1, 0] == 1.0

    raster_bin = rasterize_lines(gdf_lines, **grid, mode="binary")
    assert raster_bin.values[0, 0]
    assert raster_bin.values[0, 1]
    assert raster_bin.values[1, 0]


def test_rasterize_lines_crsless_input_accepts_explicit_output_crs(grid):
    gdf_lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 10)])], crs=None)

    with pytest.warns(UserWarning, match="Input has no CRS"):
        raster = rasterize_lines(gdf_lines, x=grid["x"], y=grid["y"], crs=CRS, mode="length")

    assert str(raster.rio.crs) == CRS


def test_rasterize_polygons(grid, grid_gdf):
    """
    Test the correctness of polygon rasterization by comparing with geopandas overlay.
    """
    # Generate random polygons
    polygons = generate_random_polygons(20, X_RANGE, Y_RANGE)
    gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=CRS)

    # Use geopandas overlay to get the expected areas
    overlay = gpd.overlay(grid_gdf, gdf_polygons.explode(index_parts=True), how="intersection")
    overlay["area"] = overlay.geometry.area
    expected_areas = overlay.groupby(["row", "col"])["area"].sum().reset_index()
    expected_areas = expected_areas.merge(grid_gdf[["row", "col"]], on=["row", "col"], how="right")
    expected_areas = expected_areas.fillna(0)["area"].values.reshape((len(Y), len(X)))

    # Rasterize with mode='area'
    raster_area = rasterize_polygons(gdf_polygons, **grid, mode="area")
    np.testing.assert_allclose(raster_area.values, expected_areas)

    # Rasterize with mode='binary' and check for consistency
    raster_bin = rasterize_polygons(gdf_polygons, **grid, mode="binary")
    expected_bin = expected_areas > 0
    np.testing.assert_array_equal(raster_bin.values, expected_bin)


def test_rasterize_polygons_geoseries_infers_crs(grid):
    gdf_polygons = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)], crs=CRS)

    raster_gdf = rasterize_polygons(gdf_polygons, x=grid["x"], y=grid["y"], mode="area")
    raster_series = rasterize_polygons(gdf_polygons.geometry, x=grid["x"], y=grid["y"], mode="area")

    np.testing.assert_allclose(raster_series.values, raster_gdf.values)
    assert str(raster_series.rio.crs) == CRS


def test_rasterize_polygons_without_any_crs_is_supported(grid):
    geometry = [box(0, 0, 10, 10), box(20, 20, 30, 30)]
    gdf_polygons = gpd.GeoDataFrame(geometry=geometry, crs=None)
    expected = rasterize_polygons(gpd.GeoDataFrame(geometry=geometry, crs=CRS), **grid, mode="area")

    raster = rasterize_polygons(gdf_polygons, x=grid["x"], y=grid["y"], mode="area")

    np.testing.assert_allclose(raster.values, expected.values)
    assert raster.rio.crs is None


def test_rasterize_polygons_rejects_non_uniform_grid(grid):
    gdf_polygons = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=CRS)
    bad_y = np.array([0.5, 1.5, 3.0], dtype=float)

    with pytest.raises(ValueError, match="y must be evenly spaced."):
        rasterize_polygons(gdf_polygons, x=grid["x"], y=bad_y, mode="area")


def test_rasterize_polygons_crsless_input_accepts_explicit_output_crs(grid):
    gdf_polygons = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=None)

    with pytest.warns(UserWarning, match="Input has no CRS"):
        raster = rasterize_polygons(gdf_polygons, x=grid["x"], y=grid["y"], crs=CRS, mode="area")

    assert str(raster.rio.crs) == CRS


def test_timing():
    x = np.arange(0.5, TIMING_GRID_SIZE, 1.0, dtype=float)
    y = np.arange(0.5, TIMING_GRID_SIZE, 1.0, dtype=float)
    polygons = generate_timing_polygons(TIMING_POLYGON_COUNT, TIMING_GRID_SIZE, np.random.default_rng(0))

    raster = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        raster = rasterize_polygons(polygons, x=x, y=y, crs=CRS, mode="area")
        elapsed = time.perf_counter() - start
        print(f"{elapsed:.2f} s")

    assert raster is not None
    assert raster.shape == (len(y), len(x))


def test_rasterize_polygons_with_weight(grid, grid_gdf):
    """
    Test polygon rasterization with a weight column.
    """
    # Generate random polygons
    polygons = generate_random_polygons(20, X_RANGE, Y_RANGE)
    gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=CRS)
    gdf_polygons["weight"] = np.random.rand(len(gdf_polygons)) * 10
    gdf_polygons["__polygon_area"] = gdf_polygons.area

    # Use geopandas overlay to get the expected weighted areas
    overlay = gpd.overlay(grid_gdf, gdf_polygons.explode(index_parts=True), how="intersection")
    overlay["area"] = overlay.geometry.area
    # The weight is in the right geodataframe, which is the second one
    overlay["weighted_area"] = overlay.area * overlay.weight / overlay["__polygon_area"]
    expected_weighted_areas = overlay.groupby(["row", "col"])["weighted_area"].sum().reset_index()
    expected_weighted_areas = expected_weighted_areas.merge(grid_gdf[["row", "col"]], on=["row", "col"], how="right")
    expected_weighted_areas = expected_weighted_areas.fillna(0)["weighted_area"].values.reshape((len(Y), len(X)))

    # Rasterize with mode='area' and weight
    raster_weighted = rasterize_polygons(gdf_polygons, **grid, mode="area", weight="weight")
    np.testing.assert_allclose(raster_weighted.values, expected_weighted_areas)


def test_rasterize_polygons_weight_errors(grid):
    """
    Test error handling for the weight argument in polygon rasterization.
    """
    polygons = generate_random_polygons(5, X_RANGE, Y_RANGE)
    gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=CRS)
    gdf_polygons["weight"] = np.random.rand(len(gdf_polygons))
    gdf_polygons["non_numeric_weight"] = ["a", "b", "c", "d", "e"]

    with pytest.raises(ValueError, match="Weight argument is not supported for binary mode."):
        rasterize_polygons(gdf_polygons, **grid, mode="binary", weight="weight")

    with pytest.raises(ValueError, match="Weight column 'non_existent_column' not found in GeoDataFrame."):
        rasterize_polygons(gdf_polygons, **grid, mode="area", weight="non_existent_column")

    with pytest.raises(ValueError, match="Weight column 'non_numeric_weight' must be numeric."):
        rasterize_polygons(gdf_polygons, **grid, mode="area", weight="non_numeric_weight")

    with pytest.raises(ValueError, match="Weight argument requires a GeoDataFrame input."):
        rasterize_polygons(gdf_polygons.geometry, x=grid["x"], y=grid["y"], mode="area", weight="weight")


def _make_donut_polygon():
    shell = Point(50.0, 50.0).buffer(30.0, quad_segs=32)
    hole = Point(50.0, 50.0).buffer(10.0, quad_segs=32)
    return Polygon(shell.exterior.coords, [hole.exterior.coords])


def _force_polygon_strategy(monkeypatch, strategy):
    polygon_module = importlib.import_module("rasterizer.polygons")
    threshold = 10**9 if strategy == "direct" else 0
    monkeypatch.setattr(polygon_module, "_HYBRID_POLYGON_THRESHOLD_CELLS", threshold)


def _expected_polygon_areas(grid_gdf, gdf_polygons):
    overlay = gpd.overlay(grid_gdf, gdf_polygons, how="intersection", keep_geom_type=False)
    overlay["area"] = overlay.geometry.area
    expected = overlay.groupby(["row", "col"])["area"].sum().reset_index()
    expected = expected.merge(grid_gdf[["row", "col"]], on=["row", "col"], how="right")
    return expected.fillna(0)["area"].values.reshape((len(Y), len(X)))


@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
@pytest.mark.parametrize(
    ("mode", "expected_transform"),
    [
        ("area", lambda expected: expected),
        ("binary", lambda expected: expected > 0),
    ],
)
def test_rasterize_polygon_with_hole_matches_geopandas(
    grid,
    grid_gdf,
    monkeypatch,
    strategy,
    mode,
    expected_transform,
):
    """
    Check donut polygons against GeoPandas on both direct and hybrid paths.
    """
    _force_polygon_strategy(monkeypatch, strategy)
    gdf_polygons = gpd.GeoDataFrame(geometry=[_make_donut_polygon()], crs=CRS)

    expected = expected_transform(_expected_polygon_areas(grid_gdf, gdf_polygons))
    raster = rasterize_polygons(gdf_polygons, **grid, mode=mode)

    if mode == "area":
        np.testing.assert_allclose(raster.values, expected)
    else:
        np.testing.assert_array_equal(raster.values, expected)


def test_rasterize_large_polygon_hybrid_matches_exact(grid, monkeypatch):
    """
    Force both polygon engines and check they produce identical values.
    """
    gdf_polygons = gpd.GeoDataFrame(geometry=[_make_donut_polygon()], crs=CRS)

    _force_polygon_strategy(monkeypatch, "direct")
    raster_exact = rasterize_polygons(gdf_polygons, **grid, mode="area")

    _force_polygon_strategy(monkeypatch, "hybrid")
    raster_hybrid = rasterize_polygons(gdf_polygons, **grid, mode="area")

    np.testing.assert_allclose(raster_hybrid.values, raster_exact.values)


def test_rasterize_grid_aligned_polygon_hybrid_matches_exact(grid, monkeypatch):
    """
    Grid-aligned boundaries need adjacent boundary cells marked before interior fill.
    """
    gdf_polygons = gpd.GeoDataFrame(geometry=[box(10, 10, 70, 70)], crs=CRS)

    _force_polygon_strategy(monkeypatch, "direct")
    raster_exact = rasterize_polygons(gdf_polygons, **grid, mode="area")

    _force_polygon_strategy(monkeypatch, "hybrid")
    raster_hybrid = rasterize_polygons(gdf_polygons, **grid, mode="area")

    np.testing.assert_allclose(raster_hybrid.values, raster_exact.values)


@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
def test_rasterize_large_polygon_with_hole_and_weight(grid, grid_gdf, monkeypatch, strategy):
    """
    Check weighted donut polygons against GeoPandas on both direct and hybrid paths.
    """
    _force_polygon_strategy(monkeypatch, strategy)
    gdf_polygons = gpd.GeoDataFrame({"weight": [7.5]}, geometry=[_make_donut_polygon()], crs=CRS)
    gdf_polygons["__polygon_area"] = gdf_polygons.area

    overlay = gpd.overlay(grid_gdf, gdf_polygons, how="intersection", keep_geom_type=False)
    overlay["area"] = overlay.geometry.area
    overlay["weighted_area"] = overlay["area"] * overlay["weight"] / overlay["__polygon_area"]
    expected = overlay.groupby(["row", "col"])["weighted_area"].sum().reset_index()
    expected = expected.merge(grid_gdf[["row", "col"]], on=["row", "col"], how="right")
    expected = expected.fillna(0)["weighted_area"].values.reshape((len(Y), len(X)))

    raster_weighted = rasterize_polygons(gdf_polygons, **grid, mode="area", weight="weight")
    np.testing.assert_allclose(raster_weighted.values, expected)


def test_rasterize_lines_with_weight(grid, grid_gdf):
    """
    Test line rasterization with a weight column.
    """
    # Generate random lines
    lines = generate_random_lines(50, X_RANGE, Y_RANGE)
    gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=CRS)
    gdf_lines["weight"] = np.random.rand(len(gdf_lines)) * 10

    gdf_lines["__line_length"] = gdf_lines.length

    # Use geopandas overlay to get the expected weighted lengths
    overlay = gpd.overlay(grid_gdf, gdf_lines.explode(index_parts=True), how="intersection", keep_geom_type=False)
    overlay["length"] = overlay.geometry.length
    # The weight is in the right geodataframe, which is the second one
    overlay["weighted_length"] = overlay.length * overlay.weight / overlay["__line_length"]
    expected_weighted_lengths = overlay.groupby(["row", "col"])["weighted_length"].sum().reset_index()
    expected_weighted_lengths = expected_weighted_lengths.merge(
        grid_gdf[["row", "col"]], on=["row", "col"], how="right"
    )
    expected_weighted_lengths = expected_weighted_lengths.fillna(0)["weighted_length"].values.reshape((len(Y), len(X)))

    # Rasterize with mode='length' and weight
    raster_weighted = rasterize_lines(gdf_lines, **grid, mode="length", weight="weight")
    np.testing.assert_allclose(raster_weighted.values, expected_weighted_lengths, atol=1e-3)


def test_rasterize_lines_weight_errors(grid):
    """
    Test error handling for the weight argument in line rasterization.
    """
    lines = generate_random_lines(5, X_RANGE, Y_RANGE)
    gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=CRS)
    gdf_lines["weight"] = np.random.rand(len(gdf_lines))
    gdf_lines["non_numeric_weight"] = ["a", "b", "c", "d", "e"][: len(gdf_lines)]

    with pytest.raises(ValueError, match="Weight argument is not supported for binary mode."):
        rasterize_lines(gdf_lines, **grid, mode="binary", weight="weight")

    with pytest.raises(ValueError, match="Weight column 'non_existent_column' not found in GeoDataFrame."):
        rasterize_lines(gdf_lines, **grid, mode="length", weight="non_existent_column")

    with pytest.raises(ValueError, match="Weight column 'non_numeric_weight' must be numeric."):
        rasterize_lines(gdf_lines, **grid, mode="length", weight="non_numeric_weight")

    with pytest.raises(ValueError, match="Weight argument requires a GeoDataFrame input."):
        rasterize_lines(gdf_lines.geometry, x=grid["x"], y=grid["y"], mode="length", weight="weight")


class _FakeProgressBar:
    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.updated = 0

    def update(self, n=1):
        self.updated += n


class _FakeProgressContext:
    def __init__(self, bar):
        self.bar = bar

    def __enter__(self):
        return self.bar

    def __exit__(self, exc_type, exc, tb):
        return False


def test_rasterize_lines_progress_bar_tracks_exploded_lines(grid, monkeypatch):
    line_module = importlib.import_module("rasterizer.lines")
    bars = []

    def fake_progress(total, desc, enabled):
        assert enabled is True
        bar = _FakeProgressBar(total=total, desc=desc)
        bars.append(bar)
        return _FakeProgressContext(bar)

    monkeypatch.setattr(line_module, "maybe_progress_bar", fake_progress)

    gdf_lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (10, 10)]),
            MultiLineString([[(10, 0), (10, 10)], [(20, 0), (20, 10)]]),
        ],
        crs=CRS,
    )

    rasterize_lines(gdf_lines, **grid, mode="length", progress_bar=True)

    assert len(bars) == 1
    assert bars[0].desc == "Rasterizing lines"
    assert bars[0].total == 3
    assert bars[0].updated == 3


def test_rasterize_polygons_progress_bar_tracks_exploded_polygons(grid, monkeypatch):
    polygon_module = importlib.import_module("rasterizer.polygons")
    bars = []

    def fake_progress(total, desc, enabled):
        assert enabled is True
        bar = _FakeProgressBar(total=total, desc=desc)
        bars.append(bar)
        return _FakeProgressContext(bar)

    monkeypatch.setattr(polygon_module, "maybe_progress_bar", fake_progress)

    gdf_polygons = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 10, 10),
            MultiPolygon([box(20, 20, 30, 30), box(40, 40, 50, 50)]),
        ],
        crs=CRS,
    )

    rasterize_polygons(gdf_polygons, **grid, mode="area", progress_bar=True)

    assert len(bars) == 1
    assert bars[0].desc == "Rasterizing polygons"
    assert bars[0].total == 3
    assert bars[0].updated == 3
