Changelog
=========

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
