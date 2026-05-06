Changelog
=========

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
