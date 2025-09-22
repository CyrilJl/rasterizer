Usage
=====

Here is a basic example of how to use `rasterizer`:

.. code-block:: python

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
