import geopandas as gpd
import numpy as np
import pytest
import rioxarray
from shapely.geometry import LineString, MultiLineString

from rasterizer import rasterize_lines

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

    with pytest.raises(ValueError, match="Le mode doit être 'binary' ou 'length'"):
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
    # The clipping algorithm might assign the full length to one side depending on floating point arithmetic.
    # A robust test should check that the total length is correct and distributed among neighbors.
    assert np.isclose(raster_len.values[5, 0] + raster_len.values[5, 1], 1.0)

    raster_bin = rasterize_lines(gdf, **grid, mode="binary")
    assert raster_bin.values[5, 0]
    assert raster_bin.values[5, 1]
