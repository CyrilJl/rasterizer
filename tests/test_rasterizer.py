import random

import geopandas as gpd
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
)

from rasterizer import rasterize_lines, rasterize_polygons

# Common setup for tests
CRS = "EPSG:32631"  # UTM 31N, metric CRS
X_RANGE = (0, 100)
Y_RANGE = (0, 100)
DX = 1.0
DY = 1.0
X = np.arange(X_RANGE[0] + DX / 2, X_RANGE[1], DX)
Y = np.arange(Y_RANGE[0] + DY / 2, Y_RANGE[1], DY)


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


def test_rasterize_polygons(grid, grid_gdf):
    """
    Test the correctness of polygon rasterization by comparing with geopandas overlay.
    """
    # Generate random polygons
    polygons = generate_random_polygons(20, X_RANGE, Y_RANGE)
    gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=CRS)

    # Use geopandas overlay to get the expected areas
    overlay = gpd.overlay(grid_gdf, gdf_polygons.explode(), how="intersection")
    overlay["area"] = overlay.geometry.area
    expected_areas = overlay.groupby(["row", "col"])["area"].sum().reset_index()
    expected_areas = expected_areas.merge(grid_gdf[["row", "col"]], on=["row", "col"], how="right")
    expected_areas = expected_areas.fillna(0)["area"].values.reshape((len(Y), len(X)))

    # Rasterize with mode='area'
    raster_area = rasterize_polygons(gdf_polygons, **grid, mode="area")
    np.testing.assert_allclose(raster_area.values, expected_areas, atol=1e-3)

    # Rasterize with mode='binary' and check for consistency
    # raster_bin = rasterize_polygons(gdf_polygons, **grid, mode="binary")
    # expected_bin = expected_areas > 0
    # np.testing.assert_array_equal(raster_bin.values, expected_bin)
