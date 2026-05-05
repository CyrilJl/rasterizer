import math

import numpy as np
from numba import njit

_GRID_EPS = 1e-12


@njit(cache=True)
def _validate_regular_axis_numba(axis: np.ndarray, rtol: float, atol: float) -> tuple[bool, int, float]:
    step = axis[1] - axis[0]
    if step <= 0.0:
        return False, 1, 0.0

    for i in range(len(axis) - 1):
        diff = axis[i + 1] - axis[i]
        if diff <= 0.0:
            return False, 1, 0.0
        if abs(diff - step) > atol + rtol * abs(step):
            return False, 2, 0.0

    return True, 0, step


@njit(cache=True)
def _bbox_indices_regular_grid(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    x0: float,
    y0: float,
    inv_dx: float,
    inv_dy: float,
    half_dx: float,
    half_dy: float,
    nx: int,
    ny: int,
) -> tuple[int, int, int, int]:
    ix_start = math.floor((xmin - half_dx - x0) * inv_dx)
    ix_end = math.ceil((xmax + half_dx - x0) * inv_dx) + 1
    iy_start = math.floor((ymin - half_dy - y0) * inv_dy)
    iy_end = math.ceil((ymax + half_dy - y0) * inv_dy) + 1

    ix_start = max(0, ix_start)
    iy_start = max(0, iy_start)
    ix_end = min(nx, ix_end)
    iy_end = min(ny, iy_end)
    return ix_start, ix_end, iy_start, iy_end


@njit(cache=True)
def _fill_indices_regular_grid(
    x_left: float,
    x_right: float,
    x0: float,
    inv_dx: float,
    ix_min: int,
    ix_max: int,
) -> tuple[int, int]:
    fill_ix_start = math.floor((x_left - x0) * inv_dx) + 1
    fill_ix_end = math.ceil((x_right - x0) * inv_dx)
    fill_ix_start = max(ix_min, fill_ix_start)
    fill_ix_end = min(ix_max, fill_ix_end)
    return fill_ix_start, fill_ix_end


@njit(cache=True)
def _ring_bounds_numba(coords: np.ndarray, start: int, end: int) -> tuple[float, float, float, float]:
    xmin = coords[start, 0]
    ymin = coords[start, 1]
    xmax = xmin
    ymax = ymin
    for i in range(start + 1, end):
        px = coords[i, 0]
        py = coords[i, 1]
        if px < xmin:
            xmin = px
        if px > xmax:
            xmax = px
        if py < ymin:
            ymin = py
        if py > ymax:
            ymax = py
    return xmin, ymin, xmax, ymax


@njit(cache=True)
def _clip_segment_to_grid_t(
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    x_grid_min: float,
    x_grid_max: float,
    y_grid_min: float,
    y_grid_max: float,
) -> tuple[bool, float, float]:
    vx = xb - xa
    vy = yb - ya
    t0 = 0.0
    t1 = 1.0

    if vx == 0.0:
        if xa < x_grid_min or xa > x_grid_max:
            return False, 0.0, 0.0
    else:
        tx0 = (x_grid_min - xa) / vx
        tx1 = (x_grid_max - xa) / vx
        if tx0 > tx1:
            tx0, tx1 = tx1, tx0
        if tx0 > t0:
            t0 = tx0
        if tx1 < t1:
            t1 = tx1

    if vy == 0.0:
        if ya < y_grid_min or ya > y_grid_max:
            return False, 0.0, 0.0
    else:
        ty0 = (y_grid_min - ya) / vy
        ty1 = (y_grid_max - ya) / vy
        if ty0 > ty1:
            ty0, ty1 = ty1, ty0
        if ty0 > t0:
            t0 = ty0
        if ty1 < t1:
            t1 = ty1

    if t1 <= t0:
        return False, 0.0, 0.0
    return True, t0, t1


@njit(cache=True)
def _next_vertical_crossing_t(
    x_pos: float,
    xa: float,
    vx: float,
    x_grid_min: float,
    x_grid_max: float,
    dx: float,
    t_current: float,
    t_end: float,
) -> float:
    if vx == 0.0:
        return t_end

    if vx > 0.0:
        line_idx = math.floor((x_pos - x_grid_min) / dx) + 1
        while line_idx * dx + x_grid_min <= x_grid_max:
            t = (x_grid_min + line_idx * dx - xa) / vx
            if t > t_current + _GRID_EPS:
                return min(t, t_end)
            line_idx += 1
    else:
        line_idx = math.ceil((x_pos - x_grid_min) / dx) - 1
        while line_idx * dx + x_grid_min >= x_grid_min:
            t = (x_grid_min + line_idx * dx - xa) / vx
            if t > t_current + _GRID_EPS:
                return min(t, t_end)
            line_idx -= 1
    return t_end


@njit(cache=True)
def _next_horizontal_crossing_t(
    y_pos: float,
    ya: float,
    vy: float,
    y_grid_min: float,
    y_grid_max: float,
    dy: float,
    t_current: float,
    t_end: float,
) -> float:
    if vy == 0.0:
        return t_end

    if vy > 0.0:
        line_idx = math.floor((y_pos - y_grid_min) / dy) + 1
        while line_idx * dy + y_grid_min <= y_grid_max:
            t = (y_grid_min + line_idx * dy - ya) / vy
            if t > t_current + _GRID_EPS:
                return min(t, t_end)
            line_idx += 1
    else:
        line_idx = math.ceil((y_pos - y_grid_min) / dy) - 1
        while line_idx * dy + y_grid_min >= y_grid_min:
            t = (y_grid_min + line_idx * dy - ya) / vy
            if t > t_current + _GRID_EPS:
                return min(t, t_end)
            line_idx -= 1
    return t_end


@njit(cache=True)
def _grid_cell_index(value: float, grid_min: float, inv_d: float) -> int:
    return math.floor((value - grid_min) * inv_d)


@njit(cache=True)
def _is_on_grid_line(value: float, grid_min: float, inv_d: float, line_count: int) -> bool:
    scaled = (value - grid_min) * inv_d
    nearest = math.floor(scaled + 0.5)
    return 0 <= nearest <= line_count and abs(scaled - nearest) <= 1e-10


@njit(cache=True)
def _mark_binary_line_cell(
    raster_data: np.ndarray,
    ix: int,
    iy: int,
    nx: int,
    ny: int,
    duplicate_left: bool,
    duplicate_down: bool,
) -> None:
    if 0 <= ix < nx and 0 <= iy < ny:
        raster_data[iy, ix] = 1
    if duplicate_left and 0 <= ix - 1 < nx and 0 <= iy < ny:
        raster_data[iy, ix - 1] = 1
    if duplicate_down and 0 <= ix < nx and 0 <= iy - 1 < ny:
        raster_data[iy - 1, ix] = 1
    if duplicate_left and duplicate_down and 0 <= ix - 1 < nx and 0 <= iy - 1 < ny:
        raster_data[iy - 1, ix - 1] = 1


@njit(cache=True)
def _mark_boundary_mask_cell(
    boundary_mask: np.ndarray,
    ix: int,
    iy: int,
    bbox_ix_start: int,
    bbox_ix_end: int,
    bbox_iy_start: int,
    bbox_iy_end: int,
    duplicate_left: bool,
    duplicate_down: bool,
) -> None:
    if bbox_ix_start <= ix < bbox_ix_end and bbox_iy_start <= iy < bbox_iy_end:
        boundary_mask[iy - bbox_iy_start, ix - bbox_ix_start] = 1
    if duplicate_left and bbox_ix_start <= ix - 1 < bbox_ix_end and bbox_iy_start <= iy < bbox_iy_end:
        boundary_mask[iy - bbox_iy_start, ix - 1 - bbox_ix_start] = 1
    if duplicate_down and bbox_ix_start <= ix < bbox_ix_end and bbox_iy_start <= iy - 1 < bbox_iy_end:
        boundary_mask[iy - 1 - bbox_iy_start, ix - bbox_ix_start] = 1
    if (
        duplicate_left
        and duplicate_down
        and bbox_ix_start <= ix - 1 < bbox_ix_end
        and bbox_iy_start <= iy - 1 < bbox_iy_end
    ):
        boundary_mask[iy - 1 - bbox_iy_start, ix - 1 - bbox_ix_start] = 1


@njit(cache=True)
def _rasterize_line_segment_traversal(
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    weight: float,
    dx: float,
    dy: float,
    x_grid_min: float,
    x_grid_max: float,
    y_grid_min: float,
    y_grid_max: float,
    mode_is_binary: bool,
    raster_data: np.ndarray,
) -> None:
    accepted, t_current, t_end = _clip_segment_to_grid_t(xa, ya, xb, yb, x_grid_min, x_grid_max, y_grid_min, y_grid_max)
    if not accepted:
        return

    vx = xb - xa
    vy = yb - ya
    segment_length = math.sqrt(vx * vx + vy * vy)
    if segment_length <= 1e-12:
        return

    nx = raster_data.shape[1]
    ny = raster_data.shape[0]
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    duplicate_left = vx == 0.0 and _is_on_grid_line(xa, x_grid_min, inv_dx, nx)
    duplicate_down = vy == 0.0 and _is_on_grid_line(ya, y_grid_min, inv_dy, ny)

    while t_current < t_end - _GRID_EPS:
        x_current = xa + vx * t_current
        y_current = ya + vy * t_current
        tx = _next_vertical_crossing_t(x_current, xa, vx, x_grid_min, x_grid_max, dx, t_current, t_end)
        ty = _next_horizontal_crossing_t(y_current, ya, vy, y_grid_min, y_grid_max, dy, t_current, t_end)
        t_next = min(tx, ty, t_end)

        if t_next <= t_current:
            break

        t_mid = 0.5 * (t_current + t_next)
        x_mid = xa + vx * t_mid
        y_mid = ya + vy * t_mid
        ix = _grid_cell_index(x_mid, x_grid_min, inv_dx)
        iy = _grid_cell_index(y_mid, y_grid_min, inv_dy)

        if mode_is_binary:
            _mark_binary_line_cell(raster_data, ix, iy, nx, ny, duplicate_left, duplicate_down)
        elif 0 <= ix < nx and 0 <= iy < ny:
            raster_data[iy, ix] += segment_length * (t_next - t_current) * weight

        t_current = t_next


@njit(cache=True)
def _mark_boundary_segment_traversal(
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    dx: float,
    dy: float,
    x_grid_min: float,
    x_grid_max: float,
    y_grid_min: float,
    y_grid_max: float,
    bbox_ix_start: int,
    bbox_ix_end: int,
    bbox_iy_start: int,
    bbox_iy_end: int,
    nx: int,
    ny: int,
    boundary_mask: np.ndarray,
) -> None:
    accepted, t_current, t_end = _clip_segment_to_grid_t(xa, ya, xb, yb, x_grid_min, x_grid_max, y_grid_min, y_grid_max)
    if not accepted:
        return

    vx = xb - xa
    vy = yb - ya
    if vx * vx + vy * vy <= 1e-24:
        return

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    duplicate_left = vx == 0.0 and _is_on_grid_line(xa, x_grid_min, inv_dx, nx)
    duplicate_down = vy == 0.0 and _is_on_grid_line(ya, y_grid_min, inv_dy, ny)

    while t_current < t_end - _GRID_EPS:
        x_current = xa + vx * t_current
        y_current = ya + vy * t_current
        tx = _next_vertical_crossing_t(x_current, xa, vx, x_grid_min, x_grid_max, dx, t_current, t_end)
        ty = _next_horizontal_crossing_t(y_current, ya, vy, y_grid_min, y_grid_max, dy, t_current, t_end)
        t_next = min(tx, ty, t_end)

        if t_next <= t_current:
            break

        t_mid = 0.5 * (t_current + t_next)
        x_mid = xa + vx * t_mid
        y_mid = ya + vy * t_mid
        ix = _grid_cell_index(x_mid, x_grid_min, inv_dx)
        iy = _grid_cell_index(y_mid, y_grid_min, inv_dy)

        _mark_boundary_mask_cell(
            boundary_mask,
            ix,
            iy,
            bbox_ix_start,
            bbox_ix_end,
            bbox_iy_start,
            bbox_iy_end,
            duplicate_left,
            duplicate_down,
        )

        t_current = t_next


@njit(cache=True)
def _rasterize_lines_range_engine(
    geoms: np.ndarray,
    line_offsets: np.ndarray,
    line_weights: np.ndarray,
    start_line_idx: int,
    end_line_idx: int,
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
    raster_data: np.ndarray,
) -> None:
    for line_idx in range(start_line_idx, end_line_idx):
        weight = line_weights[line_idx]
        coord_start = line_offsets[line_idx]
        coord_end = line_offsets[line_idx + 1]

        for i in range(coord_start, coord_end - 1):
            xa, ya = geoms[i, 1], geoms[i, 2]
            xb, yb = geoms[i + 1, 1], geoms[i + 1, 2]

            seg_xmin, seg_xmax = min(xa, xb), max(xa, xb)
            seg_ymin, seg_ymax = min(ya, yb), max(ya, yb)

            if seg_xmax < x_grid_min or seg_xmin > x_grid_max or seg_ymax < y_grid_min or seg_ymin > y_grid_max:
                continue

            _rasterize_line_segment_traversal(
                xa,
                ya,
                xb,
                yb,
                weight,
                dx,
                dy,
                x_grid_min,
                x_grid_max,
                y_grid_min,
                y_grid_max,
                mode_is_binary,
                raster_data,
            )


@njit(cache=True)
def _rasterize_lines_engine(
    geoms: np.ndarray,
    line_offsets: np.ndarray,
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
    _rasterize_lines_range_engine(
        geoms,
        line_offsets,
        line_weights,
        0,
        len(line_weights),
        x,
        y,
        dx,
        dy,
        half_dx,
        half_dy,
        x_grid_min,
        x_grid_max,
        y_grid_min,
        y_grid_max,
        mode_is_binary,
        raster_data,
    )

    return raster_data


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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

    dx = 2.0 * half_dx
    dy = 2.0 * half_dy
    x_grid_min = x[0] - half_dx
    x_grid_max = x[-1] + half_dx
    y_grid_min = y[0] - half_dy
    y_grid_max = y[-1] + half_dy
    nx = len(x)
    ny = len(y)

    for i in range(num_coords):
        xa, ya = ring_coords[i]
        xb, yb = ring_coords[(i + 1) % num_coords]

        seg_xmin, seg_xmax = min(xa, xb), max(xa, xb)
        seg_ymin, seg_ymax = min(ya, yb), max(ya, yb)

        if seg_xmax < x_grid_min or seg_xmin > x_grid_max or seg_ymax < y_grid_min or seg_ymin > y_grid_max:
            continue

        _mark_boundary_segment_traversal(
            xa,
            ya,
            xb,
            yb,
            dx,
            dy,
            x_grid_min,
            x_grid_max,
            y_grid_min,
            y_grid_max,
            bbox_ix_start,
            bbox_ix_end,
            bbox_iy_start,
            bbox_iy_end,
            nx,
            ny,
            boundary_mask,
        )


@njit(cache=True)
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


@njit(cache=True)
def _sort_intersections_in_place(intersections: np.ndarray, count: int) -> None:
    if count == 2:
        if intersections[0] > intersections[1]:
            intersections[0], intersections[1] = intersections[1], intersections[0]
        return

    for i in range(1, count):
        value = intersections[i]
        j = i - 1
        while j >= 0 and intersections[j] > value:
            intersections[j + 1] = intersections[j]
            j -= 1
        intersections[j + 1] = value


@njit(cache=True)
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


@njit(cache=True)
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
    x0 = x[0]
    inv_dx = 1.0 / (2.0 * half_dx)
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

        _sort_intersections_in_place(intersections, count)
        local_iy = iy - iy_start
        for k in range(0, count - 1, 2):
            x_left = intersections[k]
            x_right = intersections[k + 1]
            if x_right <= x_left:
                continue

            fill_ix_start, fill_ix_end = _fill_indices_regular_grid(x_left, x_right, x0, inv_dx, ix_start, ix_end)

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


@njit(cache=True)
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
    x0 = x[0]
    y0 = y[0]
    inv_dx = 1.0 / (2.0 * half_dx)
    inv_dy = 1.0 / (2.0 * half_dy)
    nx = len(x)
    ny = len(y)
    for i in range(num_polygons):
        weight = weights[i]
        ext_start, ext_end = exteriors_offsets[i], exteriors_offsets[i + 1]
        poly_xmin, poly_ymin, poly_xmax, poly_ymax = _ring_bounds_numba(exteriors_coords, ext_start, ext_end)

        if poly_xmax < x_grid_min or poly_xmin > x_grid_max or poly_ymax < y_grid_min or poly_ymin > y_grid_max:
            continue

        ix_start, ix_end, iy_start, iy_end = _bbox_indices_regular_grid(
            poly_xmin,
            poly_ymin,
            poly_xmax,
            poly_ymax,
            x0,
            y0,
            inv_dx,
            inv_dy,
            half_dx,
            half_dy,
            nx,
            ny,
        )

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


@njit(cache=True)
def _rasterize_polygons_engine(
    start_polygon_idx: int,
    end_polygon_idx: int,
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
    _rasterize_polygons_range_engine(
        start_polygon_idx,
        end_polygon_idx,
        exteriors_coords,
        exteriors_offsets,
        interiors_coords,
        interiors_ring_offsets,
        interiors_poly_offsets,
        x,
        y,
        half_dx,
        half_dy,
        x_grid_min,
        x_grid_max,
        y_grid_min,
        y_grid_max,
        mode_is_binary,
        weights,
        large_polygon_threshold_cells,
        raster_data,
    )
    return raster_data


@njit(cache=True)
def _rasterize_polygons_range_engine(
    start_polygon_idx: int,
    end_polygon_idx: int,
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
    raster_data: np.ndarray,
) -> None:
    x0 = x[0]
    y0 = y[0]
    inv_dx = 1.0 / (2.0 * half_dx)
    inv_dy = 1.0 / (2.0 * half_dy)
    nx = len(x)
    ny = len(y)
    for i in range(start_polygon_idx, end_polygon_idx):
        weight = weights[i]
        ext_start, ext_end = exteriors_offsets[i], exteriors_offsets[i + 1]
        poly_xmin, poly_ymin, poly_xmax, poly_ymax = _ring_bounds_numba(exteriors_coords, ext_start, ext_end)

        if poly_xmax < x_grid_min or poly_xmin > x_grid_max or poly_ymax < y_grid_min or poly_ymin > y_grid_max:
            continue

        ix_start, ix_end, iy_start, iy_end = _bbox_indices_regular_grid(
            poly_xmin,
            poly_ymin,
            poly_xmax,
            poly_ymax,
            x0,
            y0,
            inv_dx,
            inv_dy,
            half_dx,
            half_dy,
            nx,
            ny,
        )

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
