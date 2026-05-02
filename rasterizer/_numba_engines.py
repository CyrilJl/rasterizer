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
def _clip_polygon_area_to_box_numba(
    subject_coords: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    scratch_a: np.ndarray,
    scratch_b: np.ndarray,
) -> float:
    """Clips a polygon to a rectangular box and returns the clipped area."""
    count = len(subject_coords)
    if count < 3:
        return 0.0

    for i in range(count):
        scratch_a[i, 0] = subject_coords[i, 0]
        scratch_a[i, 1] = subject_coords[i, 1]

    use_first_buffer = True
    for edge in range(4):
        if count == 0:
            return 0.0

        if use_first_buffer:
            src = scratch_a
            dst = scratch_b
        else:
            src = scratch_b
            dst = scratch_a

        value = xmin
        if edge == 1:
            value = xmax
        elif edge == 2:
            value = ymin
        elif edge == 3:
            value = ymax

        output_count = 0
        p1x = src[count - 1, 0]
        p1y = src[count - 1, 1]
        if edge == 0:
            p1_inside = p1x >= value
        elif edge == 1:
            p1_inside = p1x <= value
        elif edge == 2:
            p1_inside = p1y >= value
        else:
            p1_inside = p1y <= value

        for p2_idx in range(count):
            p2x = src[p2_idx, 0]
            p2y = src[p2_idx, 1]
            if edge == 0:
                p2_inside = p2x >= value
            elif edge == 1:
                p2_inside = p2x <= value
            elif edge == 2:
                p2_inside = p2y >= value
            else:
                p2_inside = p2y <= value

            if p2_inside:
                if not p1_inside:
                    if edge < 2:
                        dst[output_count, 0] = value
                        dst[output_count, 1] = p1y + (p2y - p1y) * (value - p1x) / (p2x - p1x)
                    else:
                        dst[output_count, 1] = value
                        dst[output_count, 0] = p1x + (p2x - p1x) * (value - p1y) / (p2y - p1y)
                    output_count += 1
                dst[output_count, 0] = p2x
                dst[output_count, 1] = p2y
                output_count += 1
            elif p1_inside:
                if edge < 2:
                    dst[output_count, 0] = value
                    dst[output_count, 1] = p1y + (p2y - p1y) * (value - p1x) / (p2x - p1x)
                else:
                    dst[output_count, 1] = value
                    dst[output_count, 0] = p1x + (p2x - p1x) * (value - p1y) / (p2y - p1y)
                output_count += 1

            p1x = p2x
            p1y = p2y
            p1_inside = p2_inside

        count = output_count
        use_first_buffer = not use_first_buffer

    if count < 3:
        return 0.0

    final_coords = scratch_a if use_first_buffer else scratch_b
    area = 0.0
    for i in range(count):
        j = (i + 1) % count
        area += final_coords[i, 0] * final_coords[j, 1]
        area -= final_coords[j, 0] * final_coords[i, 1]
    return abs(area) / 2.0


@numba.jit(nopython=True)
def _polygon_max_ring_vertices_numba(
    polygon_idx: int,
    exteriors_offsets: np.ndarray,
    interiors_ring_offsets: np.ndarray,
    interiors_poly_offsets: np.ndarray,
) -> int:
    max_vertices = exteriors_offsets[polygon_idx + 1] - exteriors_offsets[polygon_idx]
    poly_int_start = interiors_poly_offsets[polygon_idx]
    poly_int_end = interiors_poly_offsets[polygon_idx + 1]

    for j in range(poly_int_start, poly_int_end):
        ring_vertices = interiors_ring_offsets[j + 1] - interiors_ring_offsets[j]
        if ring_vertices > max_vertices:
            max_vertices = ring_vertices

    return max_vertices


@numba.jit(nopython=True)
def _clip_polygon_cell_area_numba(
    polygon_idx: int,
    exteriors_coords: np.ndarray,
    exteriors_offsets: np.ndarray,
    interiors_coords: np.ndarray,
    interiors_ring_offsets: np.ndarray,
    interiors_poly_offsets: np.ndarray,
    cell_xmin: float,
    cell_ymin: float,
    cell_xmax: float,
    cell_ymax: float,
    scratch_a: np.ndarray,
    scratch_b: np.ndarray,
) -> float:
    ext_start, ext_end = exteriors_offsets[polygon_idx], exteriors_offsets[polygon_idx + 1]
    exterior_coords = exteriors_coords[ext_start:ext_end]
    area = _clip_polygon_area_to_box_numba(
        exterior_coords,
        cell_xmin,
        cell_ymin,
        cell_xmax,
        cell_ymax,
        scratch_a,
        scratch_b,
    )

    poly_int_start = interiors_poly_offsets[polygon_idx]
    poly_int_end = interiors_poly_offsets[polygon_idx + 1]

    for j in range(poly_int_start, poly_int_end):
        int_start = interiors_ring_offsets[j]
        int_end = interiors_ring_offsets[j + 1]
        interior_coords = interiors_coords[int_start:int_end]
        area -= _clip_polygon_area_to_box_numba(
            interior_coords,
            cell_xmin,
            cell_ymin,
            cell_xmax,
            cell_ymax,
            scratch_a,
            scratch_b,
        )

    return area


@numba.jit(nopython=True)
def _mark_boundary_cells_for_ring(
    ring_coords: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    half_dx: float,
    half_dy: float,
    bbox_ix_start: int,
    bbox_ix_end: int,
    bbox_iy_start: int,
    bbox_iy_end: int,
    boundary_mask: np.ndarray,
) -> None:
    num_coords = len(ring_coords)
    if num_coords == 0:
        return

    for i in range(num_coords):
        xa, ya = ring_coords[i]
        xb, yb = ring_coords[(i + 1) % num_coords]

        seg_xmin, seg_xmax = min(xa, xb), max(xa, xb)
        seg_ymin, seg_ymax = min(ya, yb), max(ya, yb)

        ix_start = np.searchsorted(x, seg_xmin - half_dx, side="right") - 1
        ix_end = np.searchsorted(x, seg_xmax + half_dx, side="left") + 1
        iy_start = np.searchsorted(y, seg_ymin - half_dy, side="right") - 1
        iy_end = np.searchsorted(y, seg_ymax + half_dy, side="left") + 1

        ix_start = max(bbox_ix_start, ix_start)
        iy_start = max(bbox_iy_start, iy_start)
        ix_end = min(bbox_ix_end, ix_end)
        iy_end = min(bbox_iy_end, iy_end)

        for iy in range(iy_start, iy_end):
            cell_ymin = y[iy] - half_dy
            cell_ymax = y[iy] + half_dy
            local_iy = iy - bbox_iy_start

            for ix in range(ix_start, ix_end):
                local_ix = ix - bbox_ix_start
                if boundary_mask[local_iy, local_ix]:
                    continue

                cell_xmin = x[ix] - half_dx
                cell_xmax = x[ix] + half_dx
                clipped_length = _clip_line_cohen_sutherland_numba(
                    xa,
                    ya,
                    xb,
                    yb,
                    cell_xmin,
                    cell_ymin,
                    cell_xmax,
                    cell_ymax,
                )

                if clipped_length > 1e-12:
                    boundary_mask[local_iy, local_ix] = 1


@numba.jit(nopython=True)
def _append_scanline_intersections(
    ring_coords: np.ndarray, scan_y: float, intersections: np.ndarray, count: int
) -> int:
    num_coords = len(ring_coords)
    if num_coords == 0:
        return count

    p1 = ring_coords[num_coords - 1]
    for i in range(num_coords):
        p2 = ring_coords[i]
        y1 = p1[1]
        y2 = p2[1]

        if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
            x1 = p1[0]
            x2 = p2[0]
            intersections[count] = x1 + (scan_y - y1) * (x2 - x1) / (y2 - y1)
            count += 1

        p1 = p2

    return count


@numba.jit(nopython=True)
def _rasterize_polygon_bbox_exact(
    polygon_idx: int,
    exteriors_coords: np.ndarray,
    exteriors_offsets: np.ndarray,
    interiors_coords: np.ndarray,
    interiors_ring_offsets: np.ndarray,
    interiors_poly_offsets: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    half_dx: float,
    half_dy: float,
    mode_is_binary: bool,
    weight: float,
    ix_start: int,
    ix_end: int,
    iy_start: int,
    iy_end: int,
    raster_data: np.ndarray,
) -> None:
    max_ring_vertices = _polygon_max_ring_vertices_numba(
        polygon_idx,
        exteriors_offsets,
        interiors_ring_offsets,
        interiors_poly_offsets,
    )
    scratch_capacity = max_ring_vertices + 8
    scratch_a = np.empty((scratch_capacity, 2), dtype=np.float64)
    scratch_b = np.empty((scratch_capacity, 2), dtype=np.float64)

    for iy in range(iy_start, iy_end):
        cell_ymin = y[iy] - half_dy
        cell_ymax = y[iy] + half_dy
        for ix in range(ix_start, ix_end):
            if mode_is_binary and raster_data[iy, ix]:
                continue

            cell_xmin = x[ix] - half_dx
            cell_xmax = x[ix] + half_dx
            area = _clip_polygon_cell_area_numba(
                polygon_idx,
                exteriors_coords,
                exteriors_offsets,
                interiors_coords,
                interiors_ring_offsets,
                interiors_poly_offsets,
                cell_xmin,
                cell_ymin,
                cell_xmax,
                cell_ymax,
                scratch_a,
                scratch_b,
            )

            if area > 1e-9:
                if mode_is_binary:
                    raster_data[iy, ix] = 1
                else:
                    raster_data[iy, ix] += area * weight


@numba.jit(nopython=True)
def _rasterize_polygon_bbox_hybrid(
    polygon_idx: int,
    exteriors_coords: np.ndarray,
    exteriors_offsets: np.ndarray,
    interiors_coords: np.ndarray,
    interiors_ring_offsets: np.ndarray,
    interiors_poly_offsets: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    half_dx: float,
    half_dy: float,
    mode_is_binary: bool,
    weight: float,
    ix_start: int,
    ix_end: int,
    iy_start: int,
    iy_end: int,
    raster_data: np.ndarray,
) -> None:
    bbox_width = ix_end - ix_start
    bbox_height = iy_end - iy_start
    # Boundary cells still need exact clipping. The rest can be filled as whole
    # cells by scanline spans because the polygon boundary does not cross them.
    boundary_mask = np.zeros((bbox_height, bbox_width), dtype=np.uint8)

    ext_start, ext_end = exteriors_offsets[polygon_idx], exteriors_offsets[polygon_idx + 1]
    exterior_coords = exteriors_coords[ext_start:ext_end]
    max_ring_vertices = _polygon_max_ring_vertices_numba(
        polygon_idx,
        exteriors_offsets,
        interiors_ring_offsets,
        interiors_poly_offsets,
    )
    scratch_capacity = max_ring_vertices + 8
    scratch_a = np.empty((scratch_capacity, 2), dtype=np.float64)
    scratch_b = np.empty((scratch_capacity, 2), dtype=np.float64)

    _mark_boundary_cells_for_ring(
        exterior_coords,
        x,
        y,
        half_dx,
        half_dy,
        ix_start,
        ix_end,
        iy_start,
        iy_end,
        boundary_mask,
    )

    poly_int_start = interiors_poly_offsets[polygon_idx]
    poly_int_end = interiors_poly_offsets[polygon_idx + 1]
    total_ring_vertices = len(exterior_coords)

    for j in range(poly_int_start, poly_int_end):
        int_start = interiors_ring_offsets[j]
        int_end = interiors_ring_offsets[j + 1]
        interior_coords = interiors_coords[int_start:int_end]
        total_ring_vertices += len(interior_coords)
        _mark_boundary_cells_for_ring(
            interior_coords,
            x,
            y,
            half_dx,
            half_dy,
            ix_start,
            ix_end,
            iy_start,
            iy_end,
            boundary_mask,
        )

    full_cell_value = 1.0
    if not mode_is_binary:
        full_cell_value = 4.0 * half_dx * half_dy * weight

    intersections = np.empty(total_ring_vertices, dtype=np.float64)
    for iy in range(iy_start, iy_end):
        scan_y = y[iy]
        count = _append_scanline_intersections(exterior_coords, scan_y, intersections, 0)
        for j in range(poly_int_start, poly_int_end):
            int_start = interiors_ring_offsets[j]
            int_end = interiors_ring_offsets[j + 1]
            interior_coords = interiors_coords[int_start:int_end]
            count = _append_scanline_intersections(interior_coords, scan_y, intersections, count)

        if count < 2:
            continue

        sorted_intersections = np.sort(intersections[:count])
        local_iy = iy - iy_start
        for k in range(0, count - 1, 2):
            x_left = sorted_intersections[k]
            x_right = sorted_intersections[k + 1]
            if x_right <= x_left:
                continue

            fill_ix_start = np.searchsorted(x, x_left, side="right")
            fill_ix_end = np.searchsorted(x, x_right, side="left")
            fill_ix_start = max(ix_start, fill_ix_start)
            fill_ix_end = min(ix_end, fill_ix_end)

            for ix in range(fill_ix_start, fill_ix_end):
                local_ix = ix - ix_start
                if boundary_mask[local_iy, local_ix]:
                    continue
                if mode_is_binary:
                    raster_data[iy, ix] = 1
                else:
                    raster_data[iy, ix] += full_cell_value

    for local_iy in range(bbox_height):
        iy = iy_start + local_iy
        cell_ymin = y[iy] - half_dy
        cell_ymax = y[iy] + half_dy
        for local_ix in range(bbox_width):
            if not boundary_mask[local_iy, local_ix]:
                continue

            ix = ix_start + local_ix
            if mode_is_binary and raster_data[iy, ix]:
                continue

            cell_xmin = x[ix] - half_dx
            cell_xmax = x[ix] + half_dx
            area = _clip_polygon_cell_area_numba(
                polygon_idx,
                exteriors_coords,
                exteriors_offsets,
                interiors_coords,
                interiors_ring_offsets,
                interiors_poly_offsets,
                cell_xmin,
                cell_ymin,
                cell_xmax,
                cell_ymax,
                scratch_a,
                scratch_b,
            )

            if area > 1e-9:
                if mode_is_binary:
                    raster_data[iy, ix] = 1
                else:
                    raster_data[iy, ix] += area * weight


@numba.jit(nopython=True)
def _rasterize_polygons_exact_engine(
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

        _rasterize_polygon_bbox_exact(
            i,
            exteriors_coords,
            exteriors_offsets,
            interiors_coords,
            interiors_ring_offsets,
            interiors_poly_offsets,
            x,
            y,
            half_dx,
            half_dy,
            mode_is_binary,
            weight,
            ix_start,
            ix_end,
            iy_start,
            iy_end,
            raster_data,
        )

    return raster_data


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
    large_polygon_threshold_cells: int,
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

        bbox_cell_count = (ix_end - ix_start) * (iy_end - iy_start)
        if bbox_cell_count <= large_polygon_threshold_cells:
            _rasterize_polygon_bbox_exact(
                i,
                exteriors_coords,
                exteriors_offsets,
                interiors_coords,
                interiors_ring_offsets,
                interiors_poly_offsets,
                x,
                y,
                half_dx,
                half_dy,
                mode_is_binary,
                weight,
                ix_start,
                ix_end,
                iy_start,
                iy_end,
                raster_data,
            )
        else:
            _rasterize_polygon_bbox_hybrid(
                i,
                exteriors_coords,
                exteriors_offsets,
                interiors_coords,
                interiors_ring_offsets,
                interiors_poly_offsets,
                x,
                y,
                half_dx,
                half_dy,
                mode_is_binary,
                weight,
                ix_start,
                ix_end,
                iy_start,
                iy_end,
                raster_data,
            )
    return raster_data
