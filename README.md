# Line Rasterizer

`line-rasterizer` is a lightweight Python package for rasterizing `geopandas` GeoDataFrames containing LineString and MultiLineString geometries. It is designed to be a simple, dependency-light alternative to `gdal.RasterizeLayer`, relying on `numpy` and `xarray` for grid manipulation and a pure Python implementation of the Cohen-Sutherland algorithm for line clipping.

## Features

- Rasterize lines into a binary (presence/absence) or length-based grid.
- Works with `geopandas` GeoDataFrames.
- Outputs an `xarray.DataArray` for easy integration with other scientific Python libraries.
- No GDAL dependency for the rasterization algorithm itself.

## Installation

You can install the package directly from the source code:

```bash
pip install .
```

Make sure you have the required dependencies installed: `geopandas`, `xarray`, `numpy`, `shapely`, `rioxarray`.

## Usage

Here is a basic example of how to use `line-rasterizer`:

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from line_rasterizer import rasterize_lines

# 1. Define the output grid
crs = "EPSG:32631"  # A metric CRS (UTM 31N)
x_coords = np.arange(0.5, 100.5, 1.0)
y_coords = np.arange(0.5, 100.5, 1.0)

# 2. Create some line data
line = LineString([(10, 10), (90, 90)])
gdf = gpd.GeoDataFrame([1], geometry=[line], crs=crs)

# 3. Rasterize the lines
# Get a binary raster (True where cells are intersected)
binary_raster = rasterize_lines(gdf, x=x_coords, y=y_coords, crs=crs, mode='binary')

# Get a raster where cell values represent the length of the line within them
length_raster = rasterize_lines(gdf, x=x_coords, y=y_coords, crs=crs, mode='length')

print("Binary Raster:\n", binary_raster)
print("\nLength Raster:\n", length_raster)

# The result is an xarray.DataArray
# You can plot it easily
# length_raster.plot()
```

## How it Works

The core of the package is the `rasterize_lines` function. For each line segment in the input GeoDataFrame, it identifies the grid cells that the segment's bounding box overlaps. Then, for each of these candidate cells, it uses the **Cohen-Sutherland line clipping algorithm** to determine the portion of the segment that lies strictly inside the cell. The length of this clipped segment is then used to update the cell's value in the output raster.
