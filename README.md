# Rasterizer

`rasterizer` is a lightweight Python package for rasterizing `geopandas` GeoDataFrames containing LineString, MultiLineString, Polygon, and MultiPolygon geometries. It is designed to be a simple, dependency-light alternative to `gdal.RasterizeLayer`, relying on `numpy` and `xarray` for grid manipulation.

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

Make sure you have the required dependencies installed: `geopandas`, `xarray`, `numpy`, `shapely`, `rioxarray`.

## Usage

Here is a basic example of how to use `rasterizer`:

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from rasterizer import rasterize_lines, rasterize_polygons

# 1. Define the output grid
crs = "EPSG:32631"  # A metric CRS (UTM 31N)
x_coords = np.arange(0.5, 100.5, 1.0)
y_coords = np.arange(0.5, 100.5, 1.0)

# 2. Create some line data
line = LineString([(10, 10), (90, 90)])
gdf_lines = gpd.GeoDataFrame([1], geometry=[line], crs=crs)

# 3. Rasterize the lines
# Get a raster where cell values represent the length of the line within them
length_raster = rasterize_lines(gdf_lines, x=x_coords, y=y_coords, crs=crs)

# Get a binary raster (True where cells are intersected)
binary_raster_lines = rasterize_lines(gdf_lines, x=x_coords, y=y_coords, crs=crs, mode='binary')

print("Length Raster:\n", length_raster)
print("\nBinary Raster (Lines):\n", binary_raster_lines)

# 4. Create some polygon data
poly = Polygon([(20, 30), (80, 30), (80, 70), (20, 70)])
gdf_polygons = gpd.GeoDataFrame([1], geometry=[poly], crs=crs)

# 5. Rasterize the polygons
# Get a raster where cell values represent the area of the polygon within them
area_raster = rasterize_polygons(gdf_polygons, x=x_coords, y=y_coords, crs=crs)

# Get a binary raster (True where cells are covered)
binary_raster_polygons = rasterize_polygons(gdf_polygons, x=x_coords, y=y_coords, crs=crs, mode='binary')

print("\nArea Raster:\n", area_raster)
print("\nBinary Raster (Polygons):\n", binary_raster_polygons)


# The result is an xarray.DataArray
# You can plot it easily
# length_raster.plot()
# area_raster.plot()
```

## How it Works

The core of the package are the `rasterize_lines` and `rasterize_polygons` functions. For each geometry in the input GeoDataFrame, it identifies the grid cells that the geometry's bounding box overlaps. Then, for each of these candidate cells, it uses a clipping algorithm to determine the portion of the geometry that lies strictly inside the cell. The length or area of this clipped portion is then used to update the cell's value in the output raster.
