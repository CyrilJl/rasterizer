from typing import Any, TypeVar, cast
import warnings
from contextlib import nullcontext

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr


XRaster = TypeVar("XRaster", xr.DataArray, xr.Dataset)


def geocode(ds: XRaster, x_name: str, y_name: str, crs: Any = None) -> XRaster:
    ds.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=True)
    if crs is not None:
        ds.rio.write_crs(crs, inplace=True)
    return ds


def geometry_series(data: gpd.GeoDataFrame | gpd.GeoSeries) -> gpd.GeoSeries:
    return data.geometry if isinstance(data, gpd.GeoDataFrame) else data


def prepare_vector_input(
    data: gpd.GeoDataFrame | gpd.GeoSeries,
    crs: Any,
    allowed_geom_types: list[str],
    weight: str | None = None,
) -> tuple[gpd.GeoDataFrame | gpd.GeoSeries, Any]:
    if not isinstance(data, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise TypeError("Input must be a GeoSeries or GeoDataFrame.")

    if weight is not None:
        if isinstance(data, gpd.GeoSeries):
            raise ValueError("Weight argument requires a GeoDataFrame input.")
        if weight not in data.columns:
            raise ValueError(f"Weight column '{weight}' not found in GeoDataFrame.")
        if not np.issubdtype(np.asarray(data[weight]).dtype, np.number):
            raise ValueError(f"Weight column '{weight}' must be numeric.")

    geom = geometry_series(data).force_2d()
    if isinstance(data, gpd.GeoDataFrame):
        data = cast(gpd.GeoDataFrame, data.set_geometry(geom))
    else:
        data = geom
    data = data[geometry_series(data).geom_type.isin(allowed_geom_types)]

    resolved_crs = data.crs if crs is None else crs
    if crs is None or data.crs == crs:
        return data, resolved_crs
    if data.crs is None:
        warnings.warn(
            "Input has no CRS; assuming coordinates are already expressed in the requested output CRS.",
            UserWarning,
            stacklevel=2,
        )
        return data.set_crs(crs, allow_override=True), resolved_crs
    return data.to_crs(crs), resolved_crs


class _NoOpProgressBar:
    def update(self, n=1):
        return None


def maybe_progress_bar(total: int, desc: str, enabled: bool):
    if not enabled:
        return nullcontext(_NoOpProgressBar())

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("progress_bar=True requires tqdm to be installed.") from exc

    return tqdm(total=total, desc=desc, unit="geometry")
