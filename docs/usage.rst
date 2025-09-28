:notoc: true

Usage
=====

Visual Examples
---------------

Here are some examples of what you can do with ``rasterizer``.

Rasterizing Lines
~~~~~~~~~~~~~~~~~

You can rasterize lines in either binary or length mode, with ``x`` and ``y`` two 1D arrays, and crs anything parsable by ``pyproj``.

.. code-block:: python

   from rasterizer import rasterize_lines

   raster_binary = rasterize_lines(lines=gdf, x=x, y=y, crs=crs, mode='binary')
   raster_length = rasterize_lines(lines=gdf, x=x, y=y, crs=crs, mode='length')

.. list-table::
   :widths: 44 56

   * - **Binary Mode**
     - **Length Mode**
   * - .. image:: _static/lines_binary.png
     - .. image:: _static/lines_length.png

Rasterizing Polygons
~~~~~~~~~~~~~~~~~~~~

You can rasterize polygons in either binary or area mode.

.. code-block:: python

   from rasterizer import rasterize_polygons

   raster_binary = rasterize_polygons(lines=gdf, x=x, y=y, crs=crs, mode='binary')
   raster_area = rasterize_polygons(lines=gdf, x=x, y=y, crs=crs, mode='area')

.. list-table::
   :widths: 44 56

   * - **Binary Mode**
     - **Area Mode**
   * - .. image:: _static/polygons_binary.png
     - .. image:: _static/polygons_area.png
