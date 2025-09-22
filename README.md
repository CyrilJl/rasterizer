# Rasterizer

`rasterizer` is a lightweight Python package for rasterizing `geopandas` GeoDataFrames.

This package provides functionalities that are not present in `rasterio.features`, such as area and length-based rasterization. It is also lighter and faster than using GDAL-based solutions.

For detailed usage and API documentation, please see the [full documentation](https://rasterizer.readthedocs.io).

## Features

- Rasterize lines into a binary (presence/absence) or length-based grid.
- Rasterize polygons into a binary (presence/absence) or area-based grid.
- Works with `geopandas` GeoDataFrames.
- Outputs an `xarray.DataArray` for easy integration with other scientific Python libraries.
- No GDAL dependency for the rasterization algorithm itself.

## Installation

You can install the package directly from the source code:

```bash
pip install .
```
