import math

import geopandas as gpd
import numpy as np
import rioxarray as rio
import xarray as xr
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon


def geocode(ds: xr.DataArray, x_name: str, y_name: str, crs) -> xr.DataArray:
    """
    Géocode un xarray DataArray
    """
    ds.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=True)
    ds.rio.write_crs(crs, inplace=True)
    return ds


def clip_line_cohen_sutherland(xa, ya, xb, yb, xmin, ymin, xmax, ymax):
    """
    Implémentation optimisée utilisant l'algorithme de Cohen-Sutherland
    pour le clipping de lignes.
    """

    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def compute_outcode(x, y):
        """Calcule le code de sortie pour un point"""
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

    while True:
        if not (outcode_a | outcode_b):  # Les deux points sont à l'intérieur
            accept = True
            break

        if outcode_a & outcode_b:  # Les deux points sont à l'extérieur du même côté
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


def _polygon_area(coords):
    """
    Calcule l'aire d'un polygone en utilisant la formule de shoelace.
    Les coordonnées doivent être une liste de tuples (x, y).
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


def rasterize_lines(
    lines: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    crs,
    mode="binary",
) -> xr.DataArray:
    """
    Rastérise un GeoDataFrame de LineString et MultiLineString sur une grille régulière.

    Args:
        lines (gpd.GeoDataFrame): GeoDataFrame contenant les géométries de lignes.
        x (np.ndarray): Tableau 1D des coordonnées x des centres des mailles.
        y (np.ndarray): Tableau 1D des coordonnées y des centres des mailles.
        crs: Le système de coordonnées de référence de la grille de sortie.
        mode (str, optional): 'binary' ou 'length'. Defaults to 'binary'.
            - 'binary': la maille vaut True si elle est croisée, False sinon.
            - 'length': la maille contient la longueur totale des segments de ligne.

    Returns:
        xr.DataArray: Un DataArray rastérisé.
    """
    if mode not in ["binary", "length"]:
        raise ValueError("Le mode doit être 'binary' ou 'length'")

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

                if (
                    seg_xmax < x_grid_min
                    or seg_xmin > x_grid_max
                    or seg_ymax < y_grid_min
                    or seg_ymin > y_grid_max
                ):
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


def rasterize_polygons(
    polygons: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    crs,
    mode="binary",
) -> xr.DataArray:
    """
    Rastérise un GeoDataFrame de Polygon et MultiPolygon sur une grille régulière.

    Args:
        polygons (gpd.GeoDataFrame): GeoDataFrame contenant les géométries de polygones.
        x (np.ndarray): Tableau 1D des coordonnées x des centres des mailles.
        y (np.ndarray): Tableau 1D des coordonnées y des centres des mailles.
        crs: Le système de coordonnées de référence de la grille de sortie.
        mode (str, optional): 'binary' ou 'area'. Defaults to 'binary'.
            - 'binary': la maille vaut True si elle est couverte, False sinon.
            - 'area': la maille contient la surface du polygone qui la couvre.

    Returns:
        xr.DataArray: Un DataArray rastérisé.
    """
    if mode not in ["binary", "area"]:
        raise ValueError("Le mode doit être 'binary' ou 'area'")

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

    for geom in polygons_proj.geometry:
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
