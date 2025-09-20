import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon
from tqdm.auto import tqdm

from .rasterizer import geocode


def _polygon_area(coords):
    """
    Calculates the area of a polygon using the shoelace formula.
    The coordinates must be a list of (x, y) tuples.
    """
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0


def _clip_polygon(subject_coords, clip_box):
    """
    Clips a polygon using the Sutherland-Hodgman algorithm against a rectangular box.
    `subject_coords` is a list of (x, y) tuples.
    `clip_box` is (xmin, ymin, xmax, ymax).
    """
    xmin, ymin, xmax, ymax = clip_box
    clipped_coords = list(subject_coords)

    # Helper to clip against one edge of the clip box
    def clip_edge(coords, edge, value):
        # edge: 0 for left, 1 for right, 2 for bottom, 3 for top
        output = []
        if not coords:
            return output

        p1 = coords[-1]
        for p2 in coords:
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
                        output.append((ix, iy))
                    else:  # horizontal edge (bottom/top)
                        iy = value
                        ix = p1[0] + (p2[0] - p1[0]) * (value - p1[1]) / (p2[1] - p1[1])
                        output.append((ix, iy))
                output.append(p2)
            elif p1_inside:  # p1 inside, p2 outside -> intersection
                # calculate intersection
                if edge < 2:  # vertical edge
                    ix = value
                    iy = p1[1] + (p2[1] - p1[1]) * (value - p1[0]) / (p2[0] - p1[0])
                    output.append((ix, iy))
                else:  # horizontal edge
                    iy = value
                    ix = p1[0] + (p2[0] - p1[0]) * (value - p1[1]) / (p2[1] - p1[1])
                    output.append((ix, iy))
            p1 = p2
        return output

    clipped_coords = clip_edge(clipped_coords, 0, xmin)  # left
    clipped_coords = clip_edge(clipped_coords, 1, xmax)  # right
    clipped_coords = clip_edge(clipped_coords, 2, ymin)  # bottom
    clipped_coords = clip_edge(clipped_coords, 3, ymax)  # top

    return clipped_coords


def rasterize_polygons(
    polygons: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    crs,
    mode="binary",
    progress_bar=False,
) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame of Polygon and MultiPolygon on a regular grid.

    Args:
        polygons (gpd.GeoDataFrame): GeoDataFrame containing the polygon geometries.
        x (np.ndarray): 1D array of x-coordinates of the cell centers.
        y (np.ndarray): 1D array of y-coordinates of the cell centers.
        crs: The coordinate reference system of the output grid.
        mode (str, optional): 'binary' or 'area'. Defaults to 'binary'.
            - 'binary': the cell is True if covered, False otherwise.
            - 'area': the cell contains the area of the polygon that covers it.

    Returns:
        xr.DataArray: A rasterized DataArray.
    """
    if mode not in ["binary", "area"]:
        raise ValueError("Mode must be 'binary' or 'area'")

    polygons_proj = polygons.to_crs(crs)

    if mode == "binary":
        raster_data = np.full((len(y), len(x)), False, dtype=bool)
    else:
        raster_data = np.zeros((len(y), len(x)), dtype=np.float32)

    raster = xr.DataArray(raster_data, coords={"y": y, "x": x}, dims=["y", "x"])

    if polygons_proj.empty or len(x) < 2 or len(y) < 2:
        return geocode(raster, "x", "y", crs)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    x_grid_min, x_grid_max = x[0] - half_dx, x[-1] + half_dx
    y_grid_min, y_grid_max = y[0] - half_dy, y[-1] + half_dy

    for geom in tqdm(polygons_proj.geometry, disable=not progress_bar):
        geoms_to_process = []
        if isinstance(geom, MultiPolygon):
            geoms_to_process.extend(list(geom.geoms))
        elif isinstance(geom, Polygon):
            geoms_to_process.append(geom)

        for poly in geoms_to_process:
            poly_xmin, poly_ymin, poly_xmax, poly_ymax = poly.bounds

            if (
                poly_xmax < x_grid_min
                or poly_xmin > x_grid_max
                or poly_ymax < y_grid_min
                or poly_ymin > y_grid_max
            ):
                continue

            ix_start = np.searchsorted(x, poly_xmin - half_dx, side="left")
            ix_end = np.searchsorted(x, poly_xmax + half_dx, side="right")
            iy_start = np.searchsorted(y, poly_ymin - half_dy, side="left")
            iy_end = np.searchsorted(y, poly_ymax + half_dy, side="right")

            ix_start = max(0, ix_start)
            iy_start = max(0, iy_start)
            ix_end = min(len(x), ix_end)
            iy_end = min(len(y), iy_end)

            for iy in range(iy_start, iy_end):
                for ix in range(ix_start, ix_end):
                    if raster.values[iy, ix] and mode == "binary":
                        continue

                    cell_xmin = x[ix] - half_dx
                    cell_xmax = x[ix] + half_dx
                    cell_ymin = y[iy] - half_dy
                    cell_ymax = y[iy] + half_dy
                    clip_box = (cell_xmin, cell_ymin, cell_xmax, cell_ymax)

                    # Clip exterior
                    exterior_coords = list(poly.exterior.coords)
                    clipped_exterior = _clip_polygon(exterior_coords, clip_box)
                    area = _polygon_area(clipped_exterior)

                    # Clip interiors (holes) and subtract their areas
                    for interior in poly.interiors:
                        interior_coords = list(interior.coords)
                        clipped_interior = _clip_polygon(interior_coords, clip_box)
                        area -= _polygon_area(clipped_interior)

                    if area > 1e-9:
                        if mode == "binary":
                            raster.values[iy, ix] = True
                        else:  # mode == "area"
                            raster.values[iy, ix] += area

    return geocode(raster, "x", "y", crs)
