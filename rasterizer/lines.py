from typing import cast

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely import get_coordinates, get_num_geometries

from ._misc import (
    filter_to_bbox,
    geocode,
    geometry_series,
    maybe_progress_bar,
    prepare_vector_input,
    validate_regular_axis,
)
from ._numba_engines import _rasterize_lines_engine, _rasterize_lines_range_engine

_PROGRESS_CHUNK_SIZE = 128


def _explode_lines(lines: gpd.GeoDataFrame | gpd.GeoSeries) -> gpd.GeoDataFrame | gpd.GeoSeries:
    if isinstance(lines, gpd.GeoDataFrame):
        return cast(gpd.GeoDataFrame, lines.explode(index_parts=False, ignore_index=True))
    return lines.explode(index_parts=False, ignore_index=True)


def _line_coordinates_and_offsets(lines: gpd.GeoDataFrame | gpd.GeoSeries) -> tuple[np.ndarray, np.ndarray]:
    coords, geom_indexes = get_coordinates(geometry_series(lines).array, return_index=True)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    geom_indexes = np.asarray(geom_indexes, dtype=np.intp)

    coord_counts = np.bincount(geom_indexes, minlength=len(lines))
    line_offsets = np.empty(len(lines) + 1, dtype=np.intp)
    line_offsets[0] = 0
    np.cumsum(coord_counts, out=line_offsets[1:])

    return coords, line_offsets


def rasterize_lines(
    lines: gpd.GeoDataFrame | gpd.GeoSeries,
    x: np.ndarray,
    y: np.ndarray,
    crs=None,
    mode: str = "length",
    weight: str | None = None,
    progress_bar: bool = False,
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of LineString and MultiLineString on a regular,
    axis-aligned rectangular grid.

    Args:
        lines (gpd.GeoDataFrame | gpd.GeoSeries): Geospatial vector data
            containing the line geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers, with
            constant spacing.
        y (np.ndarray): 1D array of y-coordinates of the cell centers, with
            constant spacing.
        crs: The coordinate reference system of the output grid. If None,
            infer it from ``lines`` when available.
        mode (str, optional): 'binary' or 'length'. Defaults to 'length'.
            - 'binary': the cell is True if crossed, False otherwise.
            - 'length': the cell contains the total length of the line segments.
        weight (str, optional): If specified, must be a str designating a
            numerical column of the processed gdf. The computed values of the
            raster are the fraction of the length of the intersected line by
            each mesh multiplied by the value of the specified column.
            Defaults to None.
        progress_bar (bool, optional): If True, display a ``tqdm`` progress bar
            while processing exploded line geometries. Defaults to False.


    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "length"]:
        raise ValueError("Mode must be 'binary' or 'length'")

    if weight is not None:
        if mode == "binary":
            raise ValueError("Weight argument is not supported for binary mode.")
    lines_proj, crs = prepare_vector_input(lines, crs, ["LineString", "MultiLineString"], weight=weight)

    if len(x) < 2 or len(y) < 2:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float32)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    dx = validate_regular_axis(x, "x")
    dy = validate_regular_axis(y, "y")
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    x_grid_min, x_grid_max = x[0] - half_dx, x[-1] + half_dx
    y_grid_min, y_grid_max = y[0] - half_dy, y[-1] + half_dy

    lines_proj = filter_to_bbox(lines_proj, x_grid_min, y_grid_min, x_grid_max, y_grid_max)

    line_weights = None
    if weight is not None:
        # Weighted mode preserves the existing normalization against the
        # processed line length after geometric clipping.
        lines_proj = lines_proj.clip([x_grid_min, y_grid_min, x_grid_max, y_grid_max])
        lines_proj = lines_proj[lines_proj.length > 0]
        lines_proj = cast(gpd.GeoDataFrame | gpd.GeoSeries, lines_proj)
        line_lengths = geometry_series(lines_proj).length.to_numpy(dtype=np.float64)
        line_values = cast(gpd.GeoDataFrame, lines_proj)[weight].to_numpy(dtype=np.float64)
        explode_counts = np.asarray(get_num_geometries(geometry_series(lines_proj).array), dtype=np.intp)
        line_weights = np.repeat(line_values / line_lengths, explode_counts)
    elif mode != "binary":
        lines_proj = lines_proj[lines_proj.length > 0]
        lines_proj = cast(gpd.GeoDataFrame | gpd.GeoSeries, lines_proj)

    if lines_proj.empty:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float32)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    lines_proj = _explode_lines(lines_proj)
    num_lines = len(lines_proj)

    if weight is not None:
        weights = cast(np.ndarray, line_weights)
    else:
        weights = np.ones(num_lines, dtype=np.float64)

    coords, line_offsets = _line_coordinates_and_offsets(lines_proj)

    if not progress_bar:
        raster_data_float = _rasterize_lines_engine(
            coords,
            line_offsets,
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
    else:
        raster_data_float = np.zeros((len(y), len(x)), dtype=np.float64)
        with maybe_progress_bar(num_lines, "Rasterizing lines", progress_bar) as progress:
            for start_idx in range(0, num_lines, _PROGRESS_CHUNK_SIZE):
                end_idx = min(start_idx + _PROGRESS_CHUNK_SIZE, num_lines)
                _rasterize_lines_range_engine(
                    coords,
                    line_offsets,
                    weights,
                    start_idx,
                    end_idx,
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
                    raster_data_float,
                )
                progress.update(end_idx - start_idx)

    if mode == "binary":
        raster_data = raster_data_float.astype(bool)
    else:
        raster_data = raster_data_float

    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])

    return geocode(raster, "x", "y", crs)
