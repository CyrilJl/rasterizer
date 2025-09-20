import math

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString, MultiLineString

from .rasterizer import geocode


def clip_line_cohen_sutherland(xa, ya, xb, yb, xmin, ymin, xmax, ymax):
    """
    Optimized implementation using the Cohen-Sutherland algorithm
    for line clipping.
    """

    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def compute_outcode(x, y):
        """Calculates the outcode for a point."""
        code = INSIDE
        if x < xmin:
            code |= LEFT
        elif x > xmax:
            code |= RIGHT
        if y < ymin:
            code |= BOTTOM
        elif y > ymax:
            code |= TOP
        return code

    outcode_a = compute_outcode(xa, ya)
    outcode_b = compute_outcode(xb, yb)

    x1, y1 = xa, ya
    x2, y2 = xb, yb

    accept = False
    while True:
        if not (outcode_a | outcode_b):  # Both points are inside
            accept = True
            break

        if outcode_a & outcode_b:  # Both points are outside on the same side
            accept = False
            break

        outcode_out = outcode_a if outcode_a else outcode_b

        if outcode_out & TOP:
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y = ymax
        elif outcode_out & BOTTOM:
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y = ymin
        elif outcode_out & RIGHT:
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x = xmax
        elif outcode_out & LEFT:
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x = xmin

        if outcode_out == outcode_a:
            x1, y1 = x, y
            outcode_a = compute_outcode(x1, y1)
        else:
            x2, y2 = x, y
            outcode_b = compute_outcode(x2, y2)

    if accept:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    else:
        return 0.0


def rasterize_lines(
    lines: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    crs,
    mode="binary",
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of LineString and MultiLineString on a regular grid.

    Args:
        lines (gpd.GeoDataFrame): GeoDataFrame containing the line geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers.
        y (np.ndarray): 1D array of y-coordinates of the cell centers.
        crs: The coordinate reference system of the output grid.
        mode (str, optional): 'binary' or 'length'. Defaults to 'binary'.
            - 'binary': the cell is True if crossed, False otherwise.
            - 'length': the cell contains the total length of the line segments.

    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "length"]:
        raise ValueError("Mode must be 'binary' or 'length'")

    lines_proj = lines.to_crs(crs)

    if mode == "binary":
        raster_data = np.full((len(y), len(x)), False, dtype=bool)
    else:
        raster_data = np.zeros((len(y), len(x)), dtype=np.float32)

    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])

    if lines_proj.empty or len(x) < 2 or len(y) < 2:
        return geocode(raster, "x", "y", crs)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    x_grid_min, x_grid_max = x[0] - half_dx, x[-1] + half_dx
    y_grid_min, y_grid_max = y[0] - half_dy, y[-1] + half_dy

    for geom in lines_proj.geometry:
        geoms_to_process = []
        if isinstance(geom, MultiLineString):
            geoms_to_process.extend(list(geom.geoms))
        elif isinstance(geom, LineString):
            geoms_to_process.append(geom)

        for line in geoms_to_process:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                xa, ya = coords[i]
                xb, yb = coords[i + 1]

                seg_xmin, seg_xmax = min(xa, xb), max(xa, xb)
                seg_ymin, seg_ymax = min(ya, yb), max(ya, yb)

                if seg_xmax < x_grid_min or seg_xmin > x_grid_max or seg_ymax < y_grid_min or seg_ymin > y_grid_max:
                    continue

                ix_start = np.searchsorted(x, seg_xmin - half_dx, side="left")
                ix_end = np.searchsorted(x, seg_xmax + half_dx, side="right")
                iy_start = np.searchsorted(y, seg_ymin - half_dy, side="left")
                iy_end = np.searchsorted(y, seg_ymax + half_dy, side="right")

                ix_start = max(0, ix_start)
                iy_start = max(0, iy_start)
                ix_end = min(len(x), ix_end)
                iy_end = min(len(y), iy_end)

                for iy in range(iy_start, iy_end):
                    for ix in range(ix_start, ix_end):
                        cell_xmin = x[ix] - half_dx
                        cell_xmax = x[ix] + half_dx
                        cell_ymin = y[iy] - half_dy
                        cell_ymax = y[iy] + half_dy

                        clip_box_xmin, clip_box_ymin, clip_box_xmax, clip_box_ymax = (
                            cell_xmin,
                            cell_ymin,
                            cell_xmax,
                            cell_ymax,
                        )
                        if mode == "length":
                            # Implement a top-left rule by shrinking the clip box slightly
                            # to make right and top boundaries exclusive. This avoids
                            # double-counting lengths for lines on boundaries.
                            clip_box_xmax -= 1e-9
                            clip_box_ymax -= 1e-9

                        clipped_length = clip_line_cohen_sutherland(
                            xa,
                            ya,
                            xb,
                            yb,
                            clip_box_xmin,
                            clip_box_ymin,
                            clip_box_xmax,
                            clip_box_ymax,
                        )

                        if clipped_length > 1e-9:
                            if mode == "binary":
                                raster.values[iy, ix] = True
                            else:
                                raster.values[iy, ix] += clipped_length
    return geocode(raster, "x", "y", crs)
