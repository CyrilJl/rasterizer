import geopandas as gpd
import numpy as np
import xarray as xr

from .numba_impl import _rasterize_lines_engine
from .rasterizer import geocode


def rasterize_lines(
    lines: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    crs,
    mode: str = "binary",
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of LineString and MultiLineString on a regular grid.

    Args:
        lines (gpd.GeoDataFrame): GeoDataFrame containing the line geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers.
        y (np.ndarray): 1D array of y-coordinates of the cell centers.
        crs: The coordinate reference system of the output grid.
        mode (str, optional): 'binary' or 'length'. Defaults to 'binary'.
            - 'binary': the cell is True if crossed, False otherwise.
            - 'length': the cell contains the total length of the line segments.

    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "length"]:
        raise ValueError("Mode must be 'binary' or 'length'")

    lines_proj = lines.to_crs(crs)

    if lines_proj.empty or len(x) < 2 or len(y) < 2:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float32)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    x_grid_min, x_grid_max = x[0] - half_dx, x[-1] + half_dx
    y_grid_min, y_grid_max = y[0] - half_dy, y[-1] + half_dy

    geoms_to_process = (
        lines_proj.explode(index_parts=True)
        .reset_index(drop=True)
        .get_coordinates()
        .reset_index()
        .values
    )

    raster_data_float = _rasterize_lines_engine(
        geoms_to_process,
        x,
        y,
        dx,
        dy,
        half_dx,
        half_dy,
        x_grid_min,
        x_grid_max,
        y_grid_min,
        y_grid_max,
        mode == "binary",
    )

    if mode == "binary":
        raster_data = raster_data_float.astype(bool)
    else:
        raster_data = raster_data_float

    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])

    return geocode(raster, "x", "y", crs)
