import geopandas as gpd
import numpy as np
import xarray as xr

from .numba_impl import _rasterize_polygons_engine
from .rasterizer import geocode


def compute_exterior(gdf_poly: gpd.GeoDataFrame) -> np.ndarray:
    """
    Computes the exterior coordinates of a GeoDataFrame of polygons.
    """
    return gdf_poly.explode().geometry.exterior.get_coordinates().reset_index().values


def compute_interiors(gdf_poly: gpd.GeoDataFrame) -> np.ndarray:
    """
    Computes the interior coordinates of a GeoDataFrame of polygons.
    """
    # this is much faster than naively exploding all interiors
    gdf_interiors = gdf_poly[gdf_poly.geometry.count_interior_rings() > 0]
    if gdf_interiors.empty:
        return np.empty((0, 4), dtype=np.float64)

    interiors = gdf_interiors.geometry.interiors
    ret = interiors.explode(ignore_index=False).dropna().rename("geometry").reset_index()
    if ret.empty:
        return np.empty((0, 4), dtype=np.float64)

    temp_df = ret.reset_index()
    temp_df["sub_index"] = ret.groupby("index").cumcount()
    ret["sub_index"] = temp_df["sub_index"].values

    ret = gpd.GeoDataFrame(geometry=ret.geometry, data=ret[["index", "sub_index"]])
    return ret.set_index(["index", "sub_index"]).get_coordinates().reset_index().values


def rasterize_polygons(
    polygons: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    crs,
    mode: str = "area",
    weight: str = None,
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of Polygon and MultiPolygon on a regular grid.

    Args:
        polygons (gpd.GeoDataFrame): GeoDataFrame containing the polygon geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers.
        y (np.ndarray): 1D array of y-coordinates of the cell centers.
        crs: The coordinate reference system of the output grid.
        mode (str, optional): 'binary' or 'area'. Defaults to 'area'.
            - 'binary': the cell is True if covered, False otherwise.
            - 'area': the cell contains the area of the polygon that covers it.
        weight (str, optional): If specified, must be a str designating a
            numerical column of the processed gdf. The computed values of the
            raster are the area of the intersected polygon by each mesh
            multiplied by the value of the specified column. Defaults to None.


    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "area"]:
        raise ValueError("Mode must be 'binary' or 'area'")

    if weight is not None:
        if mode == "binary":
            raise ValueError("Weight argument is not supported for binary mode.")
        if weight not in polygons.columns:
            raise ValueError(f"Weight column '{weight}' not found in GeoDataFrame.")
        if not np.issubdtype(polygons[weight].dtype, np.number):
            raise ValueError(f"Weight column '{weight}' must be numeric.")

    polygons = polygons.copy()
    polygons.geometry = polygons.geometry.force_2d()

    geom_types = polygons.geometry.geom_type
    polygons = polygons[geom_types.isin(["Polygon", "MultiPolygon"])]

    polygons_proj = polygons.to_crs(crs)

    if len(x) < 2 or len(y) < 2:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float64)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    x_grid_min, x_grid_max = x[0] - half_dx, x[-1] + half_dx
    y_grid_min, y_grid_max = y[0] - half_dy, y[-1] + half_dy

    polygons_proj = polygons_proj.cx[x_grid_min:x_grid_max, y_grid_min:y_grid_max]

    if mode != "binary":
        polygons_proj = polygons_proj[polygons_proj.area > 0]

    if polygons_proj.empty:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float64)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    if weight is not None:
        polygons_proj = polygons_proj.assign(__polygon_area=polygons_proj.area)

    polygons_proj = polygons_proj.explode(index_parts=False, ignore_index=True)
    num_polygons = len(polygons_proj)

    if weight is not None:
        weights = polygons_proj[weight].values / polygons_proj["__polygon_area"].values
    else:
        weights = np.ones(num_polygons, dtype=np.float64)

    if num_polygons == 0:
        if mode == "binary":
            raster_data = np.full((len(y), len(x)), False, dtype=bool)
        else:
            raster_data = np.zeros((len(y), len(x)), dtype=np.float64)
        raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])
        return geocode(raster, "x", "y", crs)

    exteriors = compute_exterior(polygons_proj)
    interiors = compute_interiors(polygons_proj)

    exteriors_coords = np.ascontiguousarray(exteriors[:, 1:3]).astype(np.float64)
    ext_boundaries = np.where(exteriors[:-1, 0] != exteriors[1:, 0])[0] + 1
    exteriors_offsets = np.concatenate(([0], ext_boundaries, [exteriors.shape[0]]))

    interiors_coords = np.empty((0, 2), dtype=np.float64)
    interiors_ring_offsets = np.array([0], dtype=np.intp)
    interiors_poly_offsets = np.full(num_polygons + 1, 0, dtype=np.intp)

    if interiors.shape[0] > 0:
        interiors_coords = np.ascontiguousarray(interiors[:, 2:4]).astype(np.float64)
        int_ids = interiors[:, :2]
        int_ring_boundaries = np.where((int_ids[:-1, 0] != int_ids[1:, 0]) | (int_ids[:-1, 1] != int_ids[1:, 1]))[0] + 1
        interiors_ring_offsets = np.concatenate(([0], int_ring_boundaries, [int_ids.shape[0]]))

        int_ring_poly_idx = interiors[interiors_ring_offsets[:-1], 0].astype(np.intp)

        # Create offsets for interiors per polygon. This finds the start index
        # for each polygon's run of interior rings.
        interiors_poly_offsets = np.searchsorted(int_ring_poly_idx, np.arange(num_polygons + 1), side="left")

    raster_data_float = _rasterize_polygons_engine(
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
    )

    if mode == "binary":
        raster_data = raster_data_float.astype(bool)
    else:
        raster_data = raster_data_float

    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])

    return geocode(raster, "x", "y", crs)
