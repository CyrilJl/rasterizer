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
    mode: str = "length",
    weight: str = None,
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of LineString and MultiLineString on a regular grid.

    Args:
        lines (gpd.GeoDataFrame): GeoDataFrame containing the line geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers.
        y (np.ndarray): 1D array of y-coordinates of the cell centers.
        crs: The coordinate reference system of the output grid.
        mode (str, optional): 'binary' or 'length'. Defaults to 'length'.
            - 'binary': the cell is True if crossed, False otherwise.
            - 'length': the cell contains the total length of the line segments.
        weight (str, optional): If specified, must be a str designating a
            numerical column of the processed gdf. The computed values of the
            raster are the length of the intersected line by each mesh
            multiplied by the value of the specified column. Defaults to None.


    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "length"]:
        raise ValueError("Mode must be 'binary' or 'length'")

    if weight is not None:
        if mode == "binary":
            raise ValueError("Weight argument is not supported for binary mode.")
        if weight not in lines.columns:
            raise ValueError(f"Weight column '{weight}' not found in GeoDataFrame.")
        if not np.issubdtype(lines[weight].dtype, np.number):
            raise ValueError(f"Weight column '{weight}' must be numeric.")

    lines = lines.copy()
    lines.geometry = lines.geometry.force_2d()

    geom_types = lines.geometry.geom_type
    lines = lines[geom_types.isin(["LineString", "MultiLineString"])]

    lines_proj = lines.to_crs(crs)

    if len(x) < 2 or len(y) < 2:
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

    lines_proj = lines_proj.clip([x_grid_min, y_grid_min, x_grid_max, y_grid_max])

    if mode != "binary":
        lines_proj = lines_proj[lines_proj.length > 0]

    if lines_proj.empty:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float32)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    if weight is not None:
        # This normalization is analogous to how rasterize_polygons handles
        # area normalization. The weight is normalized by the total length of
        # the original feature (LineString or MultiLineString).
        lines_proj = lines_proj.assign(__line_length=lines_proj.length)

    lines_proj = lines_proj.explode(index_parts=False, ignore_index=True)
    num_lines = len(lines_proj)

    if weight is not None:
        weights = lines_proj[weight].values / lines_proj["__line_length"].values
    else:
        weights = np.ones(num_lines, dtype=np.float64)

    geoms_to_process = lines_proj.get_coordinates().reset_index().values.astype(np.float64)

    raster_data_float = _rasterize_lines_engine(
        geoms_to_process,
        weights,
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
