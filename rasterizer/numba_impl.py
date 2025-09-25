import math

import numba
import numpy as np


@numba.jit(nopython=True)
def _clip_line_cohen_sutherland_numba(
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> float:
    """Clips a line to a rectangular box."""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def compute_outcode(x, y):
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
        if not (outcode_a | outcode_b):
            accept = True
            break

        if outcode_a & outcode_b:
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
        else:
            x = 0.0
            y = 0.0

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


@numba.jit(nopython=True)
def _rasterize_lines_engine(
    geoms: np.ndarray,
    line_weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    half_dx: float,
    half_dy: float,
    x_grid_min: float,
    x_grid_max: float,
    y_grid_min: float,
    y_grid_max: float,
    mode_is_binary: bool,
) -> np.ndarray:
    """Rasterizes lines on a grid."""
    raster_data = np.zeros((len(y), len(x)), dtype=np.float64)
    for i in range(len(geoms) - 1):
        # Check if the current and next points belong to the same line
        if geoms[i, 0] == geoms[i + 1, 0]:
            line_idx = int(geoms[i, 0])
            weight = line_weights[line_idx]
            xa, ya = geoms[i, 1], geoms[i, 2]
            xb, yb = geoms[i + 1, 1], geoms[i + 1, 2]

            seg_xmin, seg_xmax = min(xa, xb), max(xa, xb)
            seg_ymin, seg_ymax = min(ya, yb), max(ya, yb)

            if seg_xmax < x_grid_min or seg_xmin > x_grid_max or seg_ymax < y_grid_min or seg_ymin > y_grid_max:
                continue

            ix_start = np.searchsorted(x, seg_xmin - half_dx, side="right") - 1
            ix_end = np.searchsorted(x, seg_xmax + half_dx, side="left") + 1
            iy_start = np.searchsorted(y, seg_ymin - half_dy, side="right") - 1
            iy_end = np.searchsorted(y, seg_ymax + half_dy, side="left") + 1

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
                    if not mode_is_binary:
                        # Implement a top-left rule by shrinking the clip box slightly
                        # to make right and top boundaries exclusive. This avoids
                        # double-counting lengths for lines on boundaries.
                        clip_box_xmax -= 1e-9
                        clip_box_ymax -= 1e-9

                    clipped_length = _clip_line_cohen_sutherland_numba(
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
                        if mode_is_binary:
                            raster_data[iy, ix] = 1
                        else:
                            raster_data[iy, ix] += clipped_length * weight

    return raster_data


@numba.jit(nopython=True)
def _polygon_area_numba(coords: np.ndarray) -> float:
    """Calculates the area of a polygon."""
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        area += coords[i, 0] * coords[j, 1]
        area -= coords[j, 0] * coords[i, 1]
    return abs(area) / 2.0


@numba.jit(nopython=True)
def _clip_polygon_numba(subject_coords: np.ndarray, clip_box: tuple) -> np.ndarray:
    """Clips a polygon to a rectangular box."""
    xmin, ymin, xmax, ymax = clip_box

    # Helper to clip against one edge of the clip box
    def clip_edge(coords, edge, value):
        # edge: 0 for left, 1 for right, 2 for bottom, 3 for top
        output = []
        if not len(coords):
            return np.empty((0, 2), dtype=np.float64)

        p1 = coords[-1]
        for p2_idx in range(len(coords)):
            p2 = coords[p2_idx]
            if edge == 0:  # left
                p1_inside = p1[0] >= value
                p2_inside = p2[0] >= value
            elif edge == 1:  # right
                p1_inside = p1[0] <= value
                p2_inside = p2[0] <= value
            elif edge == 2:  # bottom
                p1_inside = p1[1] >= value
                p2_inside = p2[1] >= value
            else:  # top
                p1_inside = p1[1] <= value
                p2_inside = p2[1] <= value

            if p2_inside:
                if not p1_inside:  # p1 outside, p2 inside -> intersection
                    # calculate intersection
                    if edge < 2:  # vertical edge (left/right)
                        ix = value
                        iy = p1[1] + (p2[1] - p1[1]) * (value - p1[0]) / (p2[0] - p1[0])
                        output.append(np.array([ix, iy], dtype=np.float64))
                    else:  # horizontal edge (bottom/top)
                        iy = value
                        ix = p1[0] + (p2[0] - p1[0]) * (value - p1[1]) / (p2[1] - p1[1])
                        output.append(np.array([ix, iy], dtype=np.float64))
                output.append(p2)
            elif p1_inside:  # p1 inside, p2 outside -> intersection
                # calculate intersection
                if edge < 2:  # vertical edge
                    ix = value
                    iy = p1[1] + (p2[1] - p1[1]) * (value - p1[0]) / (p2[0] - p1[0])
                    output.append(np.array([ix, iy], dtype=np.float64))
                else:  # horizontal edge
                    iy = value
                    ix = p1[0] + (p2[0] - p1[0]) * (value - p1[1]) / (p2[1] - p1[1])
                    output.append(np.array([ix, iy], dtype=np.float64))
            p1 = p2

        if not output:
            return np.empty((0, 2), dtype=np.float64)

        # Manual vstack
        res = np.empty((len(output), 2), dtype=np.float64)
        for i, arr in enumerate(output):
            res[i, 0] = arr[0]
            res[i, 1] = arr[1]
        return res

    clipped_coords = subject_coords
    clipped_coords = clip_edge(clipped_coords, 0, xmin)
    clipped_coords = clip_edge(clipped_coords, 1, xmax)
    clipped_coords = clip_edge(clipped_coords, 2, ymin)
    clipped_coords = clip_edge(clipped_coords, 3, ymax)

    return clipped_coords


@numba.jit(nopython=True)
def _rasterize_polygons_engine(
    num_polygons: int,
    exteriors_coords: np.ndarray,
    exteriors_offsets: np.ndarray,
    interiors_coords: np.ndarray,
    interiors_ring_offsets: np.ndarray,
    interiors_poly_offsets: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    half_dx: float,
    half_dy: float,
    x_grid_min: float,
    x_grid_max: float,
    y_grid_min: float,
    y_grid_max: float,
    mode_is_binary: bool,
    weights: np.ndarray,
) -> np.ndarray:
    """Rasterizes polygons on a grid."""
    raster_data = np.zeros((len(y), len(x)), dtype=np.float64)
    for i in range(num_polygons):
        weight = weights[i]
        ext_start, ext_end = exteriors_offsets[i], exteriors_offsets[i + 1]
        exterior_coords = exteriors_coords[ext_start:ext_end]

        poly_xmin, poly_ymin, poly_xmax, poly_ymax = (
            np.min(exterior_coords[:, 0]),
            np.min(exterior_coords[:, 1]),
            np.max(exterior_coords[:, 0]),
            np.max(exterior_coords[:, 1]),
        )

        if poly_xmax < x_grid_min or poly_xmin > x_grid_max or poly_ymax < y_grid_min or poly_ymin > y_grid_max:
            continue

        ix_start = np.searchsorted(x, poly_xmin - half_dx, side="right") - 1
        ix_end = np.searchsorted(x, poly_xmax + half_dx, side="left") + 1
        iy_start = np.searchsorted(y, poly_ymin - half_dy, side="right") - 1
        iy_end = np.searchsorted(y, poly_ymax + half_dy, side="left") + 1

        ix_start = max(0, ix_start)
        iy_start = max(0, iy_start)
        ix_end = min(len(x), ix_end)
        iy_end = min(len(y), iy_end)

        for iy in range(iy_start, iy_end):
            for ix in range(ix_start, ix_end):
                if mode_is_binary and raster_data[iy, ix]:
                    continue

                cell_xmin = x[ix] - half_dx
                cell_xmax = x[ix] + half_dx
                cell_ymin = y[iy] - half_dy
                cell_ymax = y[iy] + half_dy
                clip_box = (cell_xmin, cell_ymin, cell_xmax, cell_ymax)

                clipped_exterior = _clip_polygon_numba(exterior_coords, clip_box)
                area = _polygon_area_numba(clipped_exterior)

                if interiors_poly_offsets.shape[0] > 0:
                    poly_int_start = interiors_poly_offsets[i]
                    poly_int_end = interiors_poly_offsets[i + 1]

                    for j in range(poly_int_start, poly_int_end):
                        int_start = interiors_ring_offsets[j]
                        int_end = interiors_ring_offsets[j + 1]
                        interior_coords = interiors_coords[int_start:int_end]
                        clipped_interior = _clip_polygon_numba(interior_coords, clip_box)
                        area -= _polygon_area_numba(clipped_interior)

                if area > 1e-9:
                    if mode_is_binary:
                        raster_data[iy, ix] = 1
                    else:
                        raster_data[iy, ix] += area * weight
    return raster_data
