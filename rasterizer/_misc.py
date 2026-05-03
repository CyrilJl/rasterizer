from contextlib import nullcontext

import rioxarray
import xarray as xr


def geocode(ds: xr.Dataset, x_name: str, y_name: str, crs):
    ds.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=True)
    ds.rio.write_crs(crs, inplace=True)
    return ds


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
