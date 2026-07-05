from typing import cast

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely import (
    get_coordinates,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
)

from ._misc import (
    filter_to_bbox,
    geocode,
    geometry_series,
    maybe_progress_bar,
    prepare_vector_input,
    validate_regular_axis,
)
from ._numba_engines import _rasterize_polygons_range_engine

# Above this bbox size, it is cheaper to fill interior spans and clip only
# boundary cells than to clip every cell in the polygon bbox.
_HYBRID_POLYGON_THRESHOLD_CELLS = 36
_PROGRESS_CHUNK_SIZE = 128


def _geometry_array(polygons):
    if isinstance(polygons, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return geometry_series(polygons).array
    return polygons


def _polygon_all_coordinates_offsets_and_exterior_lengths(
    polygons,
    interior_coord_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geoms = _geometry_array(polygons)
    coords, geom_indexes = get_coordinates(geoms, return_index=True)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    geom_indexes = np.asarray(geom_indexes, dtype=np.intp)

    coord_counts = np.bincount(geom_indexes, minlength=len(geoms))
    offsets = np.empty(len(geoms) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(coord_counts, out=offsets[1:])

    exterior_lengths = coord_counts - interior_coord_counts
    return coords, offsets, np.asarray(exterior_lengths, dtype=np.intp)


def _repeated_ring_indexes(ring_counts: np.ndarray) -> np.ndarray:
    total_rings = int(ring_counts.sum())
    if total_rings == 0:
        return np.empty(0, dtype=np.intp)

    ring_offsets = np.empty(len(ring_counts), dtype=np.intp)
    ring_offsets[0] = 0
    np.cumsum(ring_counts[:-1], out=ring_offsets[1:])
    return np.arange(total_rings, dtype=np.intp) - np.repeat(ring_offsets, ring_counts)


def _polygon_parts_and_indexes(polygons: gpd.GeoDataFrame | gpd.GeoSeries) -> tuple[np.ndarray, np.ndarray]:
    geoms = np.asarray(geometry_series(polygons).array, dtype=object)
    parts, part_indexes = get_parts(geoms, return_index=True)
    return parts, np.asarray(part_indexes, dtype=np.intp)


def _polygon_interior_coordinates_and_offsets(
    polygons,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geoms = _geometry_array(polygons)
    ring_counts = np.asarray(get_num_interior_rings(geoms), dtype=np.intp)
    total_rings = int(ring_counts.sum())
    interior_coord_counts = np.zeros(len(geoms), dtype=np.intp)

    interiors_poly_offsets = np.empty(len(geoms) + 1, dtype=np.intp)
    interiors_poly_offsets[0] = 0
    np.cumsum(ring_counts, out=interiors_poly_offsets[1:])

    if total_rings == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.array([0], dtype=np.intp),
            interiors_poly_offsets,
            interior_coord_counts,
        )

    polygon_indexes = np.repeat(np.arange(len(geoms), dtype=np.intp), ring_counts)
    ring_indexes = _repeated_ring_indexes(ring_counts)
    rings = get_interior_ring(geoms.take(polygon_indexes), ring_indexes)
    coords, coord_ring_indexes = get_coordinates(rings, return_index=True)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    coord_ring_indexes = np.asarray(coord_ring_indexes, dtype=np.intp)

    coord_counts = np.bincount(coord_ring_indexes, minlength=total_rings)
    interiors_ring_offsets = np.empty(total_rings + 1, dtype=np.intp)
    interiors_ring_offsets[0] = 0
    np.cumsum(coord_counts, out=interiors_ring_offsets[1:])

    interior_coord_counts = np.bincount(polygon_indexes, weights=coord_counts, minlength=len(geoms)).astype(np.intp)

    return coords, interiors_ring_offsets, interiors_poly_offsets, interior_coord_counts


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

    polygon_areas = None
    if mode != "binary":
        polygon_areas = geometry_series(polygons_proj).area.to_numpy(dtype=np.float64)
        valid_area = polygon_areas > 0.0
        if not np.all(valid_area):
            polygons_proj = polygons_proj.iloc[valid_area]
            polygon_areas = polygon_areas[valid_area]
            polygons_proj = cast(gpd.GeoDataFrame | gpd.GeoSeries, polygons_proj)

    if polygons_proj.empty:
        return _empty_polygon_raster(x, y, crs, mode)

    part_geometries, part_indexes = _polygon_parts_and_indexes(polygons_proj)
    num_polygons = len(part_geometries)

    if weight is not None:
        if polygon_areas is None:
            raise RuntimeError("Internal error: weighted polygon rasterization requires polygon areas.")
        source_weights = cast(gpd.GeoDataFrame, polygons_proj)[weight].to_numpy(dtype=np.float64) / polygon_areas
        weights = source_weights[part_indexes]
    else:
        weights = np.ones(num_polygons, dtype=np.float64)

    if num_polygons == 0:
        return _empty_polygon_raster(x, y, crs, mode)

    (
        interiors_coords,
        interiors_ring_offsets,
        interiors_poly_offsets,
        interior_coord_counts,
    ) = _polygon_interior_coordinates_and_offsets(part_geometries)
    exteriors_coords, exteriors_offsets, exterior_lengths = _polygon_all_coordinates_offsets_and_exterior_lengths(
        part_geometries,
        interior_coord_counts,
    )

    # uint8 for binary keeps the accumulation buffer 8x smaller than float64
    # and can be reinterpreted as bool without a copy.
    buffer_dtype = np.uint8 if mode == "binary" else np.float64
    raster_buffer = np.zeros((len(y), len(x)), dtype=buffer_dtype)
    chunk_size = _PROGRESS_CHUNK_SIZE if progress_bar else num_polygons
    with maybe_progress_bar(num_polygons, "Rasterizing polygons", progress_bar) as progress:
        for start_idx in range(0, num_polygons, chunk_size):
            end_idx = min(start_idx + chunk_size, num_polygons)
            _rasterize_polygons_range_engine(
                start_idx,
                end_idx,
                exteriors_coords,
                exteriors_offsets,
                exterior_lengths,
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
                raster_buffer,
            )
            progress.update(end_idx - start_idx)

    raster_data = raster_buffer.view(bool) if mode == "binary" else raster_buffer

    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])

    return geocode(raster, "x", "y", crs)
