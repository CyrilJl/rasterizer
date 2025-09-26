Visual Examples
===============

Here are some examples of what you can do with `rasterizer`.

Rasterizing Lines
-----------------

You can rasterize lines in either binary or length mode.

.. list-table::
   :widths: 50 50

   * - **Binary Mode**
     - **Length Mode**
   * - .. image:: lines_binary.png
     - .. image:: lines_length.png

Rasterizing Polygons
--------------------

You can rasterize polygons in either binary or area mode.

.. list-table::
   :widths: 50 50

   * - **Binary Mode**
     - **Area Mode**
   * - .. image:: polygons_binary.png
     - .. image:: polygons_area.png
   * - .. image:: _static/lines_binary.png
     - .. image:: _static/lines_length.png

   * - .. image:: _static/polygons_binary.png
     - .. image:: _static/polygons_area.png
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


Weighted Rasterization
----------------------

You can also perform a weighted rasterization. The `weight` parameter in both `rasterize_polygons` and `rasterize_lines` can be used to specify a numerical column in the GeoDataFrame. The output raster values will be multiplied by the corresponding weight.

.. code-block:: python

   import numpy as np
   import geopandas as gpd
   from shapely.geometry import LineString, Polygon
   from rasterizer import rasterize_lines, rasterize_polygons

   # 1. Define the output grid
   crs = "EPSG:32631"  # A metric CRS (UTM 31N)
   x_coords = np.arange(0.5, 100.5, 1.0)
   y_coords = np.arange(0.5, 100.5, 1.0)

   # 2. Create some polygon data with a weight column
   poly = Polygon([(20, 30), (80, 30), (80, 70), (20, 70)])
   gdf_polygons = gpd.GeoDataFrame({'weight': [10]}, geometry=[poly], crs=crs)

   # 3. Rasterize the polygons with weights
   weighted_area_raster = rasterize_polygons(gdf_polygons, x=x_coords, y=y_coords, crs=crs, weight='weight')

   print("\nWeighted Area Raster:\n", weighted_area_raster)


   # 4. Create some line data with a weight column
   line = LineString([(10, 10), (90, 90)])
   gdf_lines = gpd.GeoDataFrame({'my_weight': [5]}, geometry=[line], crs=crs)

   # 5. Rasterize the lines with weights
   weighted_length_raster = rasterize_lines(gdf_lines, x=x_coords, y=y_coords, crs=crs, weight='my_weight')

   print("\nWeighted Length Raster:\n", weighted_length_raster)
