import geopandas as gpd
import numpy as np
import xarray as xr
from numba.core import types
from numba.typed import List
from shapely.geometry import MultiPolygon, Polygon
from tqdm.auto import tqdm

from .numba_impl import _rasterize_polygons_engine
from .rasterizer import geocode


def compute_exterior(gdf_poly):
    ret = gdf_poly.geometry.exterior.get_coordinates().reset_index().values
    return ret


def compute_interiors(gdf_poly):
    interiors = gdf_poly.geometry.interiors
    ret = interiors.explode(ignore_index=False).dropna().rename("geometry").reset_index()
    temp_df = ret.reset_index()
    temp_df["sub_index"] = ret.groupby("index").cumcount()
    ret["sub_index"] = temp_df["sub_index"].values

    ret = gpd.GeoDataFrame(geometry=ret.geometry, data=ret[["index", "sub_index"]])
    ret = ret.set_index(["index", "sub_index"]).get_coordinates().reset_index().values
    return ret


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

    polygons_proj = polygons_proj.explode(index_parts=False, ignore_index=True)

    exteriors = compute_exterior(polygons_proj)
    interiors = compute_interiors(polygons_proj)

    if exteriors.shape[0] > 0:
        ext_boundaries = np.concatenate(
            (
                [0],
                np.where(exteriors[:-1, 0] != exteriors[1:, 0])[0] + 1,
                [exteriors.shape[0]],
            )
        )

        int_ring_boundaries = None
        int_ring_poly_idx = None
        if interiors.shape[0] > 0:
            int_ids = interiors[:, :2]  # poly_idx, ring_idx
            int_ring_boundaries = np.concatenate(
                (
                    [0],
                    np.where((int_ids[:-1, 0] != int_ids[1:, 0]) | (int_ids[:-1, 1] != int_ids[1:, 1]))[0] + 1,
                    [int_ids.shape[0]],
                )
            )
            int_ring_poly_idx = interiors[int_ring_boundaries[:-1], 0]

        num_polygons = len(polygons_proj)
        int_ring_cursor = 0
        for i in tqdm(range(num_polygons), disable=not progress_bar):
            ext_start, ext_end = ext_boundaries[i], ext_boundaries[i + 1]
            exterior_coords = np.ascontiguousarray(exteriors[ext_start:ext_end, 1:3])

            interior_coords_list = List.empty_list(types.float64[:, ::1])
            if int_ring_boundaries is not None:
                while (
                    int_ring_cursor < len(int_ring_poly_idx)
                    and int_ring_poly_idx[int_ring_cursor] == i
                ):
                    int_start, int_end = (
                        int_ring_boundaries[int_ring_cursor],
                        int_ring_boundaries[int_ring_cursor + 1],
                    )
                    interior_coords = np.ascontiguousarray(
                        interiors[int_start:int_end, 2:4]
                    )
                    interior_coords_list.append(interior_coords)
                    int_ring_cursor += 1

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
