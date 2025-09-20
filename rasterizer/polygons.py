import geopandas as gpd
import numpy as np
import xarray as xr
from numba.core import types
from numba.typed import List
from shapely.geometry import MultiPolygon, Polygon
from tqdm.auto import tqdm

from .numba_impl import _rasterize_polygons_engine
from .rasterizer import geocode


def rasterize_polygons(
    polygons: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    crs,
    mode="binary",
    progress_bar=False,
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of Polygon and MultiPolygon on a regular grid.

    Args:
        polygons (gpd.GeoDataFrame): GeoDataFrame containing the polygon geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers.
        y (np.ndarray): 1D array of y-coordinates of the cell centers.
        crs: The coordinate reference system of the output grid.
        mode (str, optional): 'binary' or 'area'. Defaults to 'binary'.
            - 'binary': the cell is True if covered, False otherwise.
            - 'area': the cell contains the area of the polygon that covers it.

    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "area"]:
        raise ValueError("Mode must be 'binary' or 'area'")

    polygons_proj = polygons.to_crs(crs)

    if polygons_proj.empty or len(x) < 2 or len(y) < 2:
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

    geom_type = types.Tuple(
        (
            types.float64[:, ::1],
            types.ListType(types.float64[:, ::1]),
        )
    )
    geoms_to_process = List.empty_list(geom_type)

    for geom in tqdm(polygons_proj.geometry, disable=not progress_bar):
        geoms = []
        if isinstance(geom, MultiPolygon):
            geoms.extend(list(geom.geoms))
        elif isinstance(geom, Polygon):
            geoms.append(geom)

        for poly in geoms:
            exterior_coords = np.ascontiguousarray(poly.exterior.coords)
            interior_coords_list = List.empty_list(types.float64[:, ::1])
            for interior in poly.interiors:
                interior_coords_list.append(np.ascontiguousarray(interior.coords))
            geoms_to_process.append((exterior_coords, interior_coords_list))

    if not geoms_to_process:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float32)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    raster_data_float = _rasterize_polygons_engine(
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
