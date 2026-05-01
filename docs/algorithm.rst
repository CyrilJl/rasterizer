Algorithm Notes
===============

This page summarizes the polygon rasterization strategy used by ``rasterizer``.

Overview
--------

Polygon rasterization has two competing costs:

- exact per-cell clipping is straightforward and robust, but it becomes expensive when a polygon covers a large bounding box on the target grid
- bulk filling is much faster for polygon interiors, but it still needs exact handling near the polygon boundary

``rasterizer`` therefore uses two internal paths for polygons:

- small polygon bounding boxes use an exact engine that clips the polygon against every candidate cell
- larger polygon bounding boxes use a hybrid engine that clips only boundary cells exactly and fills the interior with scanline spans

The switch happens from an internal bbox-size threshold measured in grid cells.

Exact Path
----------

For small polygon bounding boxes, the engine iterates over candidate cells and:

- clips the polygon exterior to the cell box
- subtracts clipped interior rings when holes are present
- computes the resulting area in ``mode="area"``, or marks the cell in ``mode="binary"``

This path minimizes algorithmic complexity and is a good fit when the number of candidate cells stays modest.

Hybrid Path For Large Polygons
------------------------------

For larger polygon bounding boxes, the engine separates the work into two parts.

1. Boundary detection

Each exterior and interior ring segment is tested against overlapping cells in the polygon bounding box. Any cell crossed by the polygon boundary is marked for exact processing later.

2. Interior fill

For each scanline of cell centers, the engine computes ring intersections, sorts them, and fills the spans between intersection pairs. Cells that are not marked as boundary cells can be written directly:

- in ``mode="binary"``, filled cells become ``True``
- in ``mode="area"``, filled cells receive the full cell area, scaled by the polygon weight when a weight column is used

3. Exact boundary pass

Cells touched by the polygon boundary are then processed with the same exact clipping routine used by the small-polygon path. This preserves exact cell fractions where the geometry actually cuts through a cell.

Why This Helps
--------------

The expensive part of exact polygon rasterization is repeated clipping against many fully interior cells that all have the same answer. The hybrid strategy avoids that repeated work:

- interior cells are filled in bulk from scanline spans
- only boundary cells pay the cost of polygon clipping
- holes are handled naturally because their rings participate in both boundary marking and scanline intersections

In practice, this keeps the simple exact behavior for small cases and reduces clipping work substantially for large polygons.
