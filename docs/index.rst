.. rasterizer documentation master file, created by
   sphinx-quickstart on Mon Sep 22 15:26:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rasterizer's documentation!
======================================

`rasterizer` is a Python library for rasterizing vector data (lines and polygons) onto a regular grid. It was developed to handle rasterization based on the length or area of intersection with each grid cell, which is a feature not readily available in other libraries like Rasterio or GDAL. It is designed to be fast and memory-efficient, especially for large grids and large GeoDataFrames, using a Numba-accelerated implementation.

For example, you can rasterize polygons like this:

.. code-block:: python

    import geopandas as gpd
    import numpy as np
    from rasterizer import rasterize_polygons

    # Assuming `polygons` is your GeoDataFrame
    # and `x`, `y`, `crs` are defined for your grid
    raster = rasterize_polygons(
        polygons=gdf,
        x=x,
        y=y,
        crs=crs,
        mode='area',
        weight='your_weight_column'
    )
    # `raster` is a properly georeferenced xarray.DataArray

You can install `rasterizer` using PyPI:

.. code-block:: bash

   pip install rasterizer

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   usage
   api
