Changelog
=========

v0.3.5
------

Released 2026-07-05.

This release replaces the default large-polygon engine with a signed per-edge area accumulation path and trims polygon preprocessing memory.

- large polygons (bbox above the exact-clipping threshold) are now rasterized by accumulating signed trapezoid areas and covers per edge-cell crossing, then filling interiors with a row sweep; cost drops from ``O(vertices x boundary cells + rows x vertices)`` to ``O(edge-cell crossings + bbox)``
- the previous hybrid path is retained behind an internal vertex threshold as a fallback; forcing it remains covered by the test suite
- exact and hybrid clipping now skip interior rings whose bounding box does not touch the cell, which alone gave ~4x on a 500-polygons-with-50-holes workload
- interior ring coordinates are sliced from the single all-coordinates array instead of being extracted a second time, removing a duplicate coordinate copy and one extraction pass
- the per-cell write threshold is now relative to the cell size (``1e-12`` of one cell's area) instead of an absolute ``1e-9`` in CRS units squared; degree-scale grids no longer silently drop low-coverage boundary cells (at one arc-second per cell the old cutoff was 1.3% of a cell, and below roughly ``3e-5`` degrees per cell it exceeded the full cell area)
- clipping and accumulation arithmetic runs in cell- and window-local coordinates, so per-cell results no longer pick up rounding noise proportional to absolute coordinate magnitudes in projected CRSs
- added a regression test for the polygon clip scratch overflow fixed in ``5f7c540``, randomized cross-strategy equivalence tests (jagged stars, grid-overhanging polygons, multi-hole polygons), and degree-scale plus large-coordinate-offset grid tests

On the local review workloads the new default measured 2.8x (100k small mixed polygons) to 36x (one 16k-vertex circle) faster; the Paris building showcase (606,667 polygons, 10 m grid) went from ``6.7 s`` to ``3.9 s``.

v0.3.4
------

Released 2026-06-07.

This release improves weighted polygon rasterization throughput on large real-world building datasets by reducing Python-side geometry preparation and tightening the polygon engine data layout.

- reduced polygon preprocessing overhead by avoiding unconditional geometry copies, all-true filter copies, and full GeoPandas explode calls when multipart geometries are rare
- kept polygon expansion on Shapely arrays and only decomposes geometries that are actually multipart
- accelerated coordinate preparation by extracting polygon coordinates once and passing explicit exterior ring lengths into the Numba polygon engine
- vectorized interior-ring index preparation
- lowered the hybrid polygon threshold from ``81`` to ``36`` cells after timing the real weighted-building workload and checking aggregate output stability

On the target 15 m weighted-building workload, rasterization improved from ``17.409 s`` to a ``7.467 s`` median over three runs while preserving aggregate output statistics.

v0.3.3
------

Released 2026-05-06.

This release reduces allocation overhead in the rasterization hot paths.

- line rasterization now precomputes contiguous coordinate and offset buffers before entering the Numba kernels
- polygon rasterization uses the same precomputed coordinate layout, which avoids repeated geometry unpacking
- both line and polygon rasterization prefilter inputs against the target bounding box earlier, so fewer geometries reach the kernels
- the Numba line traversal loop now advances grid crossings incrementally instead of recomputing the next crossing from scratch on every step
- polygon rasterization reuses scratch buffers inside the exact and hybrid engines, which cuts per-geometry allocation churn

The net effect is lower memory churn and better throughput when rasterizing many geometries or larger inputs.
