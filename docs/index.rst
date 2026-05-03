.. rasterizer documentation master file, created by
   sphinx-quickstart on Mon Sep 22 15:26:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rasterizer
==========

``rasterizer`` is a Python library for rasterizing vector data (lines and polygons) onto a regular,
axis-aligned rectangular grid. It was developed to handle rasterization based on the length or area of
intersection with each grid cell, which is a feature not readily available in other libraries like Rasterio
or GDAL. It is designed to be fast and memory-efficient because it specializes in this regular-grid case,
using a Numba-accelerated implementation instead of supporting arbitrary raster layouts.

For example, you can rasterize polygons like this:

.. code-block:: python

    import geopandas as gpd
    import numpy as np
    from rasterizer import rasterize_polygons

    # Assuming `polygons` is your GeoDataFrame
    # and `x`, `y`, `crs` define a regular rectilinear grid
    raster = rasterize_polygons(
        polygons=polygons,
        x=x,
        y=y,
        crs=crs,
        mode='area',
        weight='your_weight_column', # or None
        progress_bar=False,
    )
    # `raster` is a properly georeferenced xarray.DataArray

You can install ``rasterizer`` using PyPI:

.. code-block:: bash

   pip install rasterizer

The practical tradeoff is intentional: ``rasterizer`` gains speed by assuming 1D ``x`` and ``y`` arrays of
cell centers with constant spacing, which define a regular rectilinear grid.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   usage
   large_dataset_showcase
   algorithm
   api
