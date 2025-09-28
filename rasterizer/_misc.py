import rioxarray
import xarray as xr


def geocode(ds: xr.Dataset, x_name: str, y_name: str, crs):
    ds.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=True)
    ds.rio.write_crs(crs, inplace=True)
    return ds
