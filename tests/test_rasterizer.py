import random

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from rasterizer import rasterize_lines, rasterize_polygons

# Common setup for tests
CRS = "EPSG:32631"  # UTM 31N, metric CRS
X = np.arange(0.5, 10, 1.0)  # Cell centers, dx=1
Y = np.arange(0.5, 10, 1.0)  # Cell centers, dy=1


@pytest.fixture
def grid():
    return {"x": X, "y": Y, "crs": CRS}


def test_binary_mode(grid):
    # A horizontal line crossing the grid through the middle
    line = LineString([(0, 5.5), (10, 5.5)])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs=CRS)

    raster = rasterize_lines(gdf, **grid, mode="binary")

    # The 5th row (index 5) should be all True, others False
    # y=5.5 is in the cell with center 5.5 (index 5)
    expected = np.zeros_like(raster.values, dtype=bool)
    expected[5, :] = True

    np.testing.assert_array_equal(raster.values, expected)
    assert raster.dims == ("y", "x")
    assert len(raster.x) == len(X)
    assert len(raster.y) == len(Y)
    assert str(raster.rio.crs) == CRS


def test_length_mode(grid):
    # A diagonal line exactly within one cell (cell at index 1,1)
    # Cell boundaries are x:[1,2], y:[1,2]
    line = LineString([(1.0, 1.0), (2.0, 2.0)])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs=CRS)

    raster = rasterize_lines(gdf, **grid, mode="length")

    expected = np.zeros_like(raster.values, dtype=np.float32)
    expected[1, 1] = np.sqrt(2)

    np.testing.assert_allclose(raster.values, expected, atol=1e-6)


def test_length_mode_multi_cell(grid):
    # A line crossing multiple cells
    # from (1.5, 1.5) center of cell (1,1) to (3.5, 3.5) center of cell (3,3)
    line = LineString([(1.5, 1.5), (3.5, 3.5)])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs=CRS)

    raster = rasterize_lines(gdf, **grid, mode="length")

    # The line is y=x. It crosses cells (1,1), (2,2), (3,3)
    # In cell (1,1) (x:[1,2], y:[1,2]), segment is from (1.5, 1.5) to (2,2). Length = sqrt(0.5^2+0.5^2) = sqrt(0.5)
    # In cell (2,2) (x:[2,3], y:[2,3]), segment is from (2,2) to (3,3). Length = sqrt(1^2+1^2) = sqrt(2)
    # In cell (3,3) (x:[3,4], y:[3,4]), segment is from (3,3) to (3.5, 3.5). Length = sqrt(0.5^2+0.5^2) = sqrt(0.5)
    expected = np.zeros_like(raster.values, dtype=np.float32)
    expected[1, 1] = np.sqrt(0.5)
    expected[2, 2] = np.sqrt(2.0)
    expected[3, 3] = np.sqrt(0.5)

    np.testing.assert_allclose(raster.values, expected, atol=1e-6)


def test_multilinestring(grid):
    line1 = LineString([(0, 1.5), (10, 1.5)])  # Should fill row 1
    line2 = LineString([(2.5, 0), (2.5, 10)])  # Should fill col 2
    mline = MultiLineString([line1, line2])
    gdf = gpd.GeoDataFrame([1], geometry=[mline], crs=CRS)

    raster = rasterize_lines(gdf, **grid, mode="binary")

    expected = np.zeros_like(raster.values, dtype=bool)
    expected[1, :] = True
    expected[:, 2] = True

    np.testing.assert_array_equal(raster.values, expected)


def test_empty_input(grid):
    gdf = gpd.GeoDataFrame([], geometry=[], crs=CRS)

    # Test binary mode
    raster_bin = rasterize_lines(gdf, **grid, mode="binary")
    assert not np.any(raster_bin.values)
    assert raster_bin.values.dtype == bool

    # Test length mode
    raster_len = rasterize_lines(gdf, **grid, mode="length")
    assert np.all(raster_len.values == 0)
    assert raster_len.values.dtype == np.float32


def test_no_intersection(grid):
    # A line completely outside the grid
    line = LineString([(-10, -10), (-5, -5)])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs=CRS)

    raster = rasterize_lines(gdf, **grid, mode="binary")
    assert not np.any(raster.values)


def test_invalid_mode(grid):
    line = LineString([(1, 1), (2, 2)])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs=CRS)

    with pytest.raises(ValueError, match="Mode must be 'binary' or 'length'"):
        rasterize_lines(gdf, **grid, mode="invalid_mode")


def test_line_on_boundary(grid):
    # Line along the boundary between two cells
    line = LineString([(1.0, 5.0), (1.0, 6.0)])  # Boundary between x=0 and x=1
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs=CRS)

    raster_len = rasterize_lines(gdf, **grid, mode="length")

    # The line is exactly on the boundary x=1.0, from y=5.0 to y=6.0
    # It clips to cell (y=5, x=0) and (y=5, x=1)
    # Cell (5,0) x:[0.5, 1.5], y:[4.5, 5.5]. The line is clipped from (1.0, 5.0) to (1.0, 5.5). Length=0.5
    # Cell (5,1) x:[0.5, 1.5], y:[4.5, 5.5]. The line is clipped from (1.0, 5.0) to (1.0, 5.5). Length=0.5
    # A robust test should check that the total length is correct and distributed among neighbors.
    assert np.isclose(raster_len.values[5, 0] + raster_len.values[5, 1], 1.0)

    raster_bin = rasterize_lines(gdf, **grid, mode="binary")
    assert raster_bin.values[5, 0] or raster_bin.values[5, 1]


def test_line_with_mixed_geometries(grid):
    # A mix of a line and a polygon
    line = LineString([(0, 5.5), (10, 5.5)])
    poly = Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)])
    gdf = gpd.GeoDataFrame([1, 2], geometry=[line, poly], crs=CRS)

    raster = rasterize_lines(gdf, **grid, mode="binary")

    # Only the line should be rasterized
    expected = np.zeros_like(raster.values, dtype=bool)
    expected[5, :] = True

    np.testing.assert_array_equal(raster.values, expected)


def test_line_with_z_coordinate(grid):
    # A line with a Z coordinate
    line = LineString([(0, 5.5, 10), (10, 5.5, 20)])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs=CRS)

    raster = rasterize_lines(gdf, **grid, mode="binary")

    # The Z coordinate should be ignored
    expected = np.zeros_like(raster.values, dtype=bool)
    expected[5, :] = True

    np.testing.assert_array_equal(raster.values, expected)


def test_polygon_binary_mode(grid):
    # A square polygon covering 4 cells completely
    # Cells are (1,1), (1,2), (2,1), (2,2)
    # Cell boundaries for x and y are [1,2], [2,3]
    # So the polygon should cover from 1 to 3 in both axes
    poly = Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs=CRS)

    raster = rasterize_polygons(gdf, **grid, mode="binary")

    expected = np.zeros_like(raster.values, dtype=bool)
    # y coords are 1.5, 2.5 which correspond to indices 1, 2
    # x coords are 1.5, 2.5 which correspond to indices 1, 2
    expected[1:3, 1:3] = True

    np.testing.assert_array_equal(raster.values, expected)


def test_polygon_area_mode(grid):
    # A triangle covering half of cell (1,1)
    # Cell (1,1) boundaries are x:[1,2], y:[1,2], center is (1.5, 1.5)
    poly = Polygon([(1, 1), (1, 2), (2, 1), (1, 1)])  # Area should be 0.5
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs=CRS)

    raster = rasterize_polygons(gdf, **grid, mode="area")

    expected = np.zeros_like(raster.values, dtype=np.float32)
    expected[1, 1] = 0.5

    np.testing.assert_allclose(raster.values, expected, atol=1e-6)


def test_polygon_with_hole(grid):
    # A square polygon covering cell (1,1) with a hole in the middle
    # Cell (1,1) is x:[1,2], y:[1,2]. Area is 1.0
    outer = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]
    # Hole is 0.5x0.5, area is 0.25
    inner = [(1.25, 1.25), (1.75, 1.25), (1.75, 1.75), (1.25, 1.75), (1.25, 1.25)]
    poly = Polygon(outer, [inner])
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs=CRS)

    raster = rasterize_polygons(gdf, **grid, mode="area")

    expected = np.zeros_like(raster.values, dtype=np.float32)
    expected[1, 1] = 1.0 - 0.25

    np.testing.assert_allclose(raster.values, expected, atol=1e-6)


def test_multipolygon(grid):
    # Two squares, one in cell (1,1) and one in (3,3)
    poly1 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)])
    poly2 = Polygon([(3, 3), (3, 4), (4, 4), (4, 3), (3, 3)])
    mpoly = MultiPolygon([poly1, poly2])
    gdf = gpd.GeoDataFrame([1], geometry=[mpoly], crs=CRS)

    raster = rasterize_polygons(gdf, **grid, mode="binary")

    expected = np.zeros_like(raster.values, dtype=bool)
    expected[1, 1] = True
    expected[3, 3] = True

    np.testing.assert_array_equal(raster.values, expected)


def test_polygon_empty_input(grid):
    gdf = gpd.GeoDataFrame([], geometry=[], crs=CRS)
    raster = rasterize_polygons(gdf, **grid, mode="binary")
    assert not np.any(raster.values)


def test_polygon_no_intersection(grid):
    poly = Polygon([(-1, -1), (-1, -2), (-2, -2), (-2, -1), (-1, -1)])
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs=CRS)
    raster = rasterize_polygons(gdf, **grid, mode="binary")
    assert not np.any(raster.values)


def test_polygon_invalid_mode(grid):
    poly = Polygon([(1, 1), (1, 2), (2, 1)])
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs=CRS)
    with pytest.raises(ValueError, match="Mode must be 'binary' or 'area'"):
        rasterize_polygons(gdf, **grid, mode="invalid")


def test_polygon_with_mixed_geometries(grid):
    # A mix of a polygon and a line
    poly = Polygon([(1, 1), (1, 2), (2, 1), (1, 1)])
    line = LineString([(0, 5.5), (10, 5.5)])
    gdf = gpd.GeoDataFrame([1, 2], geometry=[poly, line], crs=CRS)

    raster = rasterize_polygons(gdf, **grid, mode="binary")

    # Only the polygon should be rasterized, and it covers half of cell (1,1)
    # so in binary mode the cell is True
    expected = np.zeros_like(raster.values, dtype=bool)
    expected[1, 1] = True

    np.testing.assert_array_equal(raster.values, expected)


def test_polygon_with_z_coordinate(grid):
    # A polygon with a Z coordinate, covering half of cell (1,1)
    poly = Polygon([(1, 1, 10), (1, 2, 20), (2, 1, 40), (1, 1, 10)])
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs=CRS)

    raster = rasterize_polygons(gdf, **grid, mode="area")

    # The Z coordinate should be ignored, area is 0.5
    expected = np.zeros_like(raster.values, dtype=np.float32)
    expected[1, 1] = 0.5

    np.testing.assert_allclose(raster.values, expected, atol=1e-6)


def test_lines_concatenation_length_mode(grid):
    # Two lines in different cells
    line1 = LineString([(1.0, 1.0), (2.0, 2.0)])  # In cell (1,1)
    line2 = LineString([(3.0, 3.0), (4.0, 4.0)])  # In cell (3,3)
    gdf1 = gpd.GeoDataFrame([1], geometry=[line1], crs=CRS)
    gdf2 = gpd.GeoDataFrame([1], geometry=[line2], crs=CRS)
    gdf_concat = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

    raster1 = rasterize_lines(gdf1, **grid, mode="length")
    raster2 = rasterize_lines(gdf2, **grid, mode="length")
    raster_concat = rasterize_lines(gdf_concat, **grid, mode="length")

    np.testing.assert_allclose(raster_concat.values, raster1.values + raster2.values)


def test_lines_concatenation_binary_mode(grid):
    # Two lines in different cells
    line1 = LineString([(0, 1.5), (10, 1.5)])  # Row 1
    line2 = LineString([(2.5, 0), (2.5, 10)])  # Col 2
    gdf1 = gpd.GeoDataFrame([1], geometry=[line1], crs=CRS)
    gdf2 = gpd.GeoDataFrame([1], geometry=[line2], crs=CRS)
    gdf_concat = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

    raster1 = rasterize_lines(gdf1, **grid, mode="binary")
    raster2 = rasterize_lines(gdf2, **grid, mode="binary")
    raster_concat = rasterize_lines(gdf_concat, **grid, mode="binary")

    np.testing.assert_array_equal(raster_concat.values, np.logical_or(raster1.values, raster2.values))


def test_polygons_concatenation_area_mode(grid):
    # Two polygons in different cells
    poly1 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)])  # Cell (1,1)
    poly2 = Polygon([(3, 3), (3, 4), (4, 4), (4, 3), (3, 3)])  # Cell (3,3)
    gdf1 = gpd.GeoDataFrame([1], geometry=[poly1], crs=CRS)
    gdf2 = gpd.GeoDataFrame([1], geometry=[poly2], crs=CRS)
    gdf_concat = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

    raster1 = rasterize_polygons(gdf1, **grid, mode="area")
    raster2 = rasterize_polygons(gdf2, **grid, mode="area")
    raster_concat = rasterize_polygons(gdf_concat, **grid, mode="area")

    np.testing.assert_allclose(raster_concat.values, raster1.values + raster2.values)


def test_polygons_concatenation_binary_mode(grid):
    # Two polygons in different cells
    poly1 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)])  # Cell (1,1)
    poly2 = Polygon([(3, 3), (3, 4), (4, 4), (4, 3), (3, 3)])  # Cell (3,3)
    gdf1 = gpd.GeoDataFrame([1], geometry=[poly1], crs=CRS)
    gdf2 = gpd.GeoDataFrame([1], geometry=[poly2], crs=CRS)
    gdf_concat = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

    raster1 = rasterize_polygons(gdf1, **grid, mode="binary")
    raster2 = rasterize_polygons(gdf2, **grid, mode="binary")
    raster_concat = rasterize_polygons(gdf_concat, **grid, mode="binary")

    np.testing.assert_array_equal(raster_concat.values, np.logical_or(raster1.values, raster2.values))


# Helper functions for stress tests

def generate_random_linestrings(n, x_range, y_range, max_points=10):
    linestrings = []
    for _ in range(n):
        num_points = random.randint(2, max_points)
        points = []
        for _ in range(num_points):
            points.append(
                (random.uniform(*x_range), random.uniform(*y_range))
            )
        linestrings.append(LineString(points))
    return linestrings


def generate_random_multilinestrings(n, x_range, y_range, max_lines=5, max_points=10):
    multilinestrings = []
    for _ in range(n):
        num_lines = random.randint(1, max_lines)
        lines = []
        for _ in range(num_lines):
            num_points = random.randint(2, max_points)
            points = []
            for _ in range(num_points):
                points.append(
                    (random.uniform(*x_range), random.uniform(*y_range))
                )
            lines.append(LineString(points))
        multilinestrings.append(MultiLineString(lines))
    return multilinestrings


def generate_random_polygons(n, x_range, y_range, max_points=15, with_interiors_fraction=0.1):
    polygons = []
    for i in range(n):
        num_points = random.randint(5, max_points)
        points = np.random.rand(num_points, 2)
        points[:, 0] = points[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        points[:, 1] = points[:, 1] * (y_range[1] - y_range[0]) + y_range[0]

        try:
            hull = ConvexHull(points)
            exterior = points[hull.vertices]
            poly = Polygon(exterior)
        except Exception:
            # ConvexHull can fail with collinear points, just skip this one
            continue

        if i < n * with_interiors_fraction:
            # Create a smaller polygon for the interior
            interior_points = poly.centroid.coords[0] + (points - poly.centroid.coords[0]) * 0.5
            try:
                interior_hull = ConvexHull(interior_points)
                interior = interior_points[interior_hull.vertices]
                if Polygon(interior).is_valid:
                    poly = Polygon(exterior, [interior])
            except Exception:
                pass # It may fail, just use the exterior

        polygons.append(poly)
    return polygons


def generate_random_multipolygons(n, x_range, y_range, max_polys=5, max_points=15):
    multipolygons = []
    for _ in range(n):
        num_polys = random.randint(1, max_polys)
        polys = generate_random_polygons(num_polys, x_range, y_range, max_points, with_interiors_fraction=0.0)
        multipolygons.append(MultiPolygon(polys))
    return multipolygons

# Stress tests

def test_rasterize_lines_stress():
    n_samples_lines = 10000
    n_samples_multilines = 2500
    x_range = (0, 100)
    y_range = (0, 100)

    lines = generate_random_linestrings(n_samples_lines, x_range, y_range)
    mlines = generate_random_multilinestrings(n_samples_multilines, x_range, y_range)

    gdf = gpd.GeoDataFrame(geometry=lines + mlines, crs=CRS)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    x = np.arange(xmin, xmax, 1.0)
    y = np.arange(ymin, ymax, 1.0)

    raster_len = rasterize_lines(gdf, x=x, y=y, crs=CRS, mode="length")
    assert raster_len.sum() > 0

    raster_bin = rasterize_lines(gdf, x=x, y=y, crs=CRS, mode="binary")
    assert raster_bin.sum() > 0


def test_rasterize_polygons_stress():
    n_samples_polys = 1000
    n_samples_multipolys = 250
    x_range = (0, 100)
    y_range = (0, 100)

    polys = generate_random_polygons(n_samples_polys, x_range, y_range)
    mpolys = generate_random_multipolygons(n_samples_multipolys, x_range, y_range)

    gdf = gpd.GeoDataFrame(geometry=polys + mpolys, crs=CRS)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    x = np.arange(xmin, xmax, 1.0)
    y = np.arange(ymin, ymax, 1.0)

    raster_area = rasterize_polygons(gdf, x=x, y=y, crs=CRS, mode="area")
    assert raster_area.sum() > 0

    raster_bin = rasterize_polygons(gdf, x=x, y=y, crs=CRS, mode="binary")
    assert raster_bin.sum() > 0
