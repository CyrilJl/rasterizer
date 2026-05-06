from typing import cast

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely import get_coordinates, get_exterior_ring, get_interior_ring, get_num_interior_rings

from ._misc import (
    filter_to_bbox,
    geocode,
    geometry_series,
    maybe_progress_bar,
    prepare_vector_input,
    validate_regular_axis,
)
from ._numba_engines import _rasterize_polygons_engine, _rasterize_polygons_range_engine

# Above this bbox size, it is cheaper to fill interior spans and clip only
# boundary cells than to clip every cell in the polygon bbox.
_HYBRID_POLYGON_THRESHOLD_CELLS = 81
_PROGRESS_CHUNK_SIZE = 128


def _explode_polygons(polygons: gpd.GeoDataFrame | gpd.GeoSeries) -> gpd.GeoDataFrame | gpd.GeoSeries:
    if isinstance(polygons, gpd.GeoDataFrame):
        return cast(gpd.GeoDataFrame, polygons.explode(index_parts=False, ignore_index=True))
    return polygons.explode(index_parts=False, ignore_index=True)


def compute_exterior(polygons: gpd.GeoDataFrame | gpd.GeoSeries) -> np.ndarray:
    """
    Computes the exterior coordinates of polygons.
    """
    return geometry_series(polygons).explode().exterior.get_coordinates().reset_index().values


def compute_interiors(polygons: gpd.GeoDataFrame | gpd.GeoSeries) -> np.ndarray:
    """
    Computes the interior coordinates of polygons.
    """
    # this is much faster than naively exploding all interiors
    geom = geometry_series(polygons)
    interiors = geom[geom.count_interior_rings() > 0].interiors
    if interiors.empty:
        return np.empty((0, 4), dtype=np.float64)

    ret = interiors.explode(ignore_index=False).dropna().rename("geometry").reset_index()
    if ret.empty:
        return np.empty((0, 4), dtype=np.float64)

    temp_df = ret.reset_index()
    temp_df["sub_index"] = ret.groupby("index").cumcount()
    ret["sub_index"] = temp_df["sub_index"].values

    ret = gpd.GeoDataFrame(geometry=ret.geometry, data=ret[["index", "sub_index"]])
    return ret.set_index(["index", "sub_index"]).get_coordinates().reset_index().values


def _polygon_exterior_coordinates_and_offsets(
    polygons: gpd.GeoDataFrame | gpd.GeoSeries,
) -> tuple[np.ndarray, np.ndarray]:
    rings = get_exterior_ring(geometry_series(polygons).array)
    coords, ring_indexes = get_coordinates(rings, return_index=True)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    ring_indexes = np.asarray(ring_indexes, dtype=np.intp)

    coord_counts = np.bincount(ring_indexes, minlength=len(polygons))
    offsets = np.empty(len(polygons) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(coord_counts, out=offsets[1:])
    return coords, offsets


def _repeated_ring_indexes(ring_counts: np.ndarray) -> np.ndarray:
    total_rings = int(ring_counts.sum())
    ring_indexes = np.empty(total_rings, dtype=np.intp)
    offset = 0
    for count in ring_counts:
        end = offset + int(count)
        ring_indexes[offset:end] = np.arange(count, dtype=np.intp)
        offset = end
    return ring_indexes


def _polygon_interior_coordinates_and_offsets(
    polygons: gpd.GeoDataFrame | gpd.GeoSeries,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geoms = geometry_series(polygons).array
    ring_counts = np.asarray(get_num_interior_rings(geoms), dtype=np.intp)
    total_rings = int(ring_counts.sum())

    interiors_poly_offsets = np.empty(len(polygons) + 1, dtype=np.intp)
    interiors_poly_offsets[0] = 0
    np.cumsum(ring_counts, out=interiors_poly_offsets[1:])

    if total_rings == 0:
        return np.empty((0, 2), dtype=np.float64), np.array([0], dtype=np.intp), interiors_poly_offsets

    polygon_indexes = np.repeat(np.arange(len(polygons), dtype=np.intp), ring_counts)
    ring_indexes = _repeated_ring_indexes(ring_counts)
    rings = get_interior_ring(geoms.take(polygon_indexes), ring_indexes)
    coords, coord_ring_indexes = get_coordinates(rings, return_index=True)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    coord_ring_indexes = np.asarray(coord_ring_indexes, dtype=np.intp)

    coord_counts = np.bincount(coord_ring_indexes, minlength=total_rings)
    interiors_ring_offsets = np.empty(total_rings + 1, dtype=np.intp)
    interiors_ring_offsets[0] = 0
    np.cumsum(coord_counts, out=interiors_ring_offsets[1:])

    return coords, interiors_ring_offsets, interiors_poly_offsets


def _empty_polygon_raster(x: np.ndarray, y: np.ndarray, crs, mode: str) -> xr.DataArray:
    if mode == "binary":
        raster_data = np.full((len(y), len(x)), False, dtype=bool)
    else:
        raster_data = np.zeros((len(y), len(x)), dtype=np.float64)
    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
    return geocode(raster, "x", "y", crs)


def rasterize_polygons(
    polygons: gpd.GeoDataFrame | gpd.GeoSeries,
    x: np.ndarray,
    y: np.ndarray,
    crs=None,
    mode: str = "area",
    weight: str | None = None,
    progress_bar: bool = False,
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of Polygon and MultiPolygon on a regular,
    axis-aligned rectangular grid.

    Args:
        polygons (gpd.GeoDataFrame | gpd.GeoSeries): Geospatial vector data
            containing the polygon geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers, with
            constant spacing.
        y (np.ndarray): 1D array of y-coordinates of the cell centers, with
            constant spacing.
        crs: The coordinate reference system of the output grid. If None,
            infer it from ``polygons`` when available.
        mode (str, optional): 'binary' or 'area'. Defaults to 'area'.
            - 'binary': the cell is True if covered, False otherwise.
            - 'area': the cell contains the area of the polygon that covers it.
        weight (str, optional): If specified, must be a str designating a
            numerical column of the processed gdf. The computed values of the
            raster are the fraction of the area of the intersected polygon by
            each mesh multiplied by the value of the specified column.
            Defaults to None.
        progress_bar (bool, optional): If True, display a ``tqdm`` progress bar
            while processing exploded polygon geometries. Defaults to False.


    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "area"]:
        raise ValueError("Mode must be 'binary' or 'area'")

    if weight is not None:
        if mode == "binary":
            raise ValueError("Weight argument is not supported for binary mode.")
    polygons_proj, crs = prepare_vector_input(polygons, crs, ["Polygon", "MultiPolygon"], weight=weight)

    if len(x) < 2 or len(y) < 2:
        return _empty_polygon_raster(x, y, crs, mode)

    dx = validate_regular_axis(x, "x")
    dy = validate_regular_axis(y, "y")
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    x_grid_min, x_grid_max = x[0] - half_dx, x[-1] + half_dx
    y_grid_min, y_grid_max = y[0] - half_dy, y[-1] + half_dy

    polygons_proj = filter_to_bbox(polygons_proj, x_grid_min, y_grid_min, x_grid_max, y_grid_max)

    if mode != "binary":
        polygons_proj = polygons_proj[polygons_proj.area > 0]
        polygons_proj = cast(gpd.GeoDataFrame | gpd.GeoSeries, polygons_proj)

    if polygons_proj.empty:
        return _empty_polygon_raster(x, y, crs, mode)

    if weight is not None:
        polygons_proj = cast(gpd.GeoDataFrame, polygons_proj.assign(__polygon_area=polygons_proj.area))

    polygons_proj = _explode_polygons(polygons_proj)
    num_polygons = len(polygons_proj)

    if weight is not None:
        weights = polygons_proj[weight].values / polygons_proj["__polygon_area"].values
    else:
        weights = np.ones(num_polygons, dtype=np.float64)

    if num_polygons == 0:
        return _empty_polygon_raster(x, y, crs, mode)

    exteriors_coords, exteriors_offsets = _polygon_exterior_coordinates_and_offsets(polygons_proj)
    (
        interiors_coords,
        interiors_ring_offsets,
        interiors_poly_offsets,
    ) = _polygon_interior_coordinates_and_offsets(polygons_proj)

    if not progress_bar:
        raster_data_float = _rasterize_polygons_engine(
            0,
            num_polygons,
            exteriors_coords,
            exteriors_offsets,
            interiors_coords,
            interiors_ring_offsets,
            interiors_poly_offsets,
            x,
            y,
            half_dx,
            half_dy,
            x_grid_min,
            x_grid_max,
            y_grid_min,
            y_grid_max,
            mode == "binary",
            weights,
            _HYBRID_POLYGON_THRESHOLD_CELLS,
        )
    else:
        raster_data_float = np.zeros((len(y), len(x)), dtype=np.float64)
        with maybe_progress_bar(num_polygons, "Rasterizing polygons", progress_bar) as progress:
            for start_idx in range(0, num_polygons, _PROGRESS_CHUNK_SIZE):
                end_idx = min(start_idx + _PROGRESS_CHUNK_SIZE, num_polygons)
                _rasterize_polygons_range_engine(
                    start_idx,
                    end_idx,
                    exteriors_coords,
                    exteriors_offsets,
                    interiors_coords,
                    interiors_ring_offsets,
                    interiors_poly_offsets,
                    x,
                    y,
                    half_dx,
                    half_dy,
                    x_grid_min,
                    x_grid_max,
                    y_grid_min,
                    y_grid_max,
                    mode == "binary",
                    weights,
                    _HYBRID_POLYGON_THRESHOLD_CELLS,
                    raster_data_float,
                )
                progress.update(end_idx - start_idx)

    if mode == "binary":
        raster_data = raster_data_float.astype(bool)
    else:
        raster_data = raster_data_float

    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])

    return geocode(raster, "x", "y", crs)
