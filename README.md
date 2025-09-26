# Rasterizer

`rasterizer` is a lightweight Python package for rasterizing `geopandas` GeoDataFrames.

This package provides functionalities that are not present in `rasterio.features`, such as area and length-based rasterization. It is also lighter and faster than using GDAL-based solutions.

For detailed usage and API documentation, please see the [full documentation](https://rasterizer.readthedocs.io).

## Features

- Rasterize lines into a binary (presence/absence) or length-based grid.
- Rasterize polygons into a binary (presence/absence) or area-based grid.
- Weighted rasterization: Rasterize geometries while weighting the output by a numerical column in the GeoDataFrame.
- Works with `geopandas` GeoDataFrames.
- Outputs an `xarray.DataArray` for easy integration with other scientific Python libraries.
- No GDAL dependency for the rasterization algorithm itself.

## Installation

You can install the package directly from PyPI:

```bash
pip install rasterizer
```

## Usage

Here are some examples of what you can do with `rasterizer`.

### Rasterizing Lines

You can rasterize lines in either binary or length mode.

| Binary Mode | Length Mode |
|---|---|
| ![Lines - Binary](docs/_static/lines_binary.png) | ![Lines - Length](docs/_static/lines_length.png) |

### Rasterizing Polygons

You can rasterize polygons in either binary or area mode.

| Binary Mode | Area Mode |
|---|---|
| ![Polygons - Binary](docs/_static/polygons_binary.png) | ![Polygons - Area](docs/_static/polygons_area.png) |
