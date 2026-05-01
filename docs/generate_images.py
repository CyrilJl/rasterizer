import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, box

from rasterizer.lines import rasterize_lines
from rasterizer.polygons import rasterize_polygons

# --- Parameters ---
GRID_SIZE_X = 50  # Number of grid cells in X
GRID_SIZE_Y = 50  # Number of grid cells in Y
GRID_X_MIN = 0  # Minimum X bound
GRID_X_MAX = 5  # Maximum X bound
GRID_Y_MIN = 0  # Minimum Y bound
GRID_Y_MAX = 5  # Maximum Y bound
NUM_LINES = 13  # Number of lines to generate
NUM_POLYGONS = 15  # Number of polygons to generate
SAVE_PATH = "docs/_static"  # Path to save images


# Grid setup
x_centers = np.linspace(
    GRID_X_MIN + 0.5 * (GRID_X_MAX - GRID_X_MIN) / GRID_SIZE_X,
    GRID_X_MAX - 0.5 * (GRID_X_MAX - GRID_X_MIN) / GRID_SIZE_X,
    GRID_SIZE_X,
)
y_centers = np.linspace(
    GRID_Y_MIN + 0.5 * (GRID_Y_MAX - GRID_Y_MIN) / GRID_SIZE_Y,
    GRID_Y_MAX - 0.5 * (GRID_Y_MAX - GRID_Y_MIN) / GRID_SIZE_Y,
    GRID_SIZE_Y,
)
dx = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 1.0
dy = y_centers[1] - y_centers[0] if len(y_centers) > 1 else 1.0
x_edges = np.arange(x_centers[0] - dx / 2, x_centers[-1] + dx, dx)
y_edges = np.arange(y_centers[0] - dy / 2, y_centers[-1] + dy, dy)
crs = "EPSG:3857"


def _ring_coords_without_closure(ring) -> np.ndarray:
    return np.asarray(ring.coords[:-1], dtype=float)


def _mark_boundary_cells_for_ring_python(
    ring_coords: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    half_dx: float,
    half_dy: float,
    boundary_mask: np.ndarray,
) -> None:
    if len(ring_coords) == 0:
        return

    num_coords = len(ring_coords)
    for i in range(num_coords):
        xa, ya = ring_coords[i]
        xb, yb = ring_coords[(i + 1) % num_coords]
        segment = LineString([(xa, ya), (xb, yb)])

        seg_xmin, seg_xmax = min(xa, xb), max(xa, xb)
        seg_ymin, seg_ymax = min(ya, yb), max(ya, yb)

        ix_start = np.searchsorted(x, seg_xmin - half_dx, side="right") - 1
        ix_end = np.searchsorted(x, seg_xmax + half_dx, side="left") + 1
        iy_start = np.searchsorted(y, seg_ymin - half_dy, side="right") - 1
        iy_end = np.searchsorted(y, seg_ymax + half_dy, side="left") + 1

        ix_start = max(0, ix_start)
        iy_start = max(0, iy_start)
        ix_end = min(len(x), ix_end)
        iy_end = min(len(y), iy_end)

        for iy in range(iy_start, iy_end):
            cell_ymin = y[iy] - half_dy
            cell_ymax = y[iy] + half_dy
            for ix in range(ix_start, ix_end):
                if boundary_mask[iy, ix]:
                    continue

                cell = box(x[ix] - half_dx, cell_ymin, x[ix] + half_dx, cell_ymax)
                if segment.intersection(cell).length > 1e-12:
                    boundary_mask[iy, ix] = True


def _append_scanline_intersections_python(ring_coords: np.ndarray, scan_y: float, intersections: list[float]) -> None:
    if len(ring_coords) == 0:
        return

    p1 = ring_coords[-1]
    for p2 in ring_coords:
        y1 = p1[1]
        y2 = p2[1]
        if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
            x1 = p1[0]
            x2 = p2[0]
            intersections.append(x1 + (scan_y - y1) * (x2 - x1) / (y2 - y1))
        p1 = p2


def _compute_hybrid_demo_masks(
    polygon,
    x: np.ndarray,
    y: np.ndarray,
    half_dx: float,
    half_dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    boundary_mask = np.zeros((len(y), len(x)), dtype=bool)

    _mark_boundary_cells_for_ring_python(
        _ring_coords_without_closure(polygon.exterior),
        x,
        y,
        half_dx,
        half_dy,
        boundary_mask,
    )

    for interior in polygon.interiors:
        _mark_boundary_cells_for_ring_python(
            _ring_coords_without_closure(interior),
            x,
            y,
            half_dx,
            half_dy,
            boundary_mask,
        )

    interior_mask = np.zeros_like(boundary_mask)
    scanline_index = 0
    scanline_intersections = np.array([], dtype=float)
    max_scanline_fill = -1

    for iy, scan_y in enumerate(y):
        intersections: list[float] = []
        _append_scanline_intersections_python(
            _ring_coords_without_closure(polygon.exterior),
            scan_y,
            intersections,
        )
        for interior in polygon.interiors:
            _append_scanline_intersections_python(
                _ring_coords_without_closure(interior),
                scan_y,
                intersections,
            )

        if len(intersections) < 2:
            continue

        sorted_intersections = np.sort(np.asarray(intersections, dtype=float))
        row_fill_count = 0
        for k in range(0, len(sorted_intersections) - 1, 2):
            x_left = sorted_intersections[k]
            x_right = sorted_intersections[k + 1]
            if x_right <= x_left:
                continue

            fill_ix_start = np.searchsorted(x, x_left, side="right")
            fill_ix_end = np.searchsorted(x, x_right, side="left")
            for ix in range(fill_ix_start, fill_ix_end):
                if boundary_mask[iy, ix]:
                    continue
                interior_mask[iy, ix] = True
                row_fill_count += 1

        if row_fill_count > max_scanline_fill:
            max_scanline_fill = row_fill_count
            scanline_index = iy
            scanline_intersections = sorted_intersections

    covered_mask = np.zeros_like(boundary_mask)
    for iy, yc in enumerate(y):
        for ix, xc in enumerate(x):
            cell = box(xc - half_dx, yc - half_dy, xc + half_dx, yc + half_dy)
            if polygon.intersection(cell).area > 1e-9:
                covered_mask[iy, ix] = True

    return boundary_mask, interior_mask, covered_mask, scanline_index, scanline_intersections


def _draw_demo_grid(
    ax,
    polygon_gdf: gpd.GeoDataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
    title: str,
    note: str,
    scanline_y: float | None = None,
    scanline_intersections: np.ndarray | None = None,
    show_boundary: bool = True,
    boundary_alpha: float = 0.85,
) -> None:
    interior_color = "#60a5fa"
    boundary_color = "#fb923c"

    for iy in range(boundary_mask.shape[0]):
        for ix in range(boundary_mask.shape[1]):
            xmin = x_edges[ix]
            ymin = y_edges[iy]
            width = x_edges[ix + 1] - x_edges[ix]
            height = y_edges[iy + 1] - y_edges[iy]

            if show_boundary and boundary_mask[iy, ix]:
                ax.add_patch(
                    patches.Rectangle(
                        (xmin, ymin),
                        width,
                        height,
                        facecolor=boundary_color,
                        edgecolor="none",
                        alpha=boundary_alpha,
                        zorder=1,
                    )
                )

            if interior_mask[iy, ix]:
                ax.add_patch(
                    patches.Rectangle(
                        (xmin, ymin),
                        width,
                        height,
                        facecolor=interior_color,
                        edgecolor="none",
                        alpha=0.85,
                        zorder=2,
                    )
                )

    for x_edge in x_edges:
        ax.axvline(x_edge, color="#94a3b8", lw=1.0, zorder=3)
    for y_edge in y_edges:
        ax.axhline(y_edge, color="#94a3b8", lw=1.0, zorder=3)

    polygon_gdf.boundary.plot(ax=ax, color="#0f172a", linewidth=2.8, zorder=4)

    if scanline_y is not None:
        ax.axhline(scanline_y, color="#2563eb", lw=1.8, linestyle=(0, (6, 4)), zorder=5)
    if scanline_intersections is not None and len(scanline_intersections) > 0:
        ax.scatter(
            scanline_intersections,
            np.full(len(scanline_intersections), scanline_y),
            color="#1d4ed8",
            s=18,
            zorder=6,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.text(
        0.5,
        -0.16,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.0,
        color="#334155",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def generate_algorithm_hybrid_figure(save_path: str) -> None:
    x_demo = np.arange(6, dtype=float) + 0.5
    y_demo = np.arange(6, dtype=float) + 0.5
    half_dx_demo = 0.5
    half_dy_demo = 0.5
    x_edges_demo = np.arange(7, dtype=float)
    y_edges_demo = np.arange(7, dtype=float)

    polygon = Point(3.1, 3.0).buffer(2.25, resolution=48)
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)

    boundary_mask, interior_mask, covered_mask, scanline_index, scanline_intersections = _compute_hybrid_demo_masks(
        polygon,
        x_demo,
        y_demo,
        half_dx_demo,
        half_dy_demo,
    )
    exact_boundary_mask = boundary_mask & covered_mask

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.28, wspace=0.24)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle("Hybrid Path: Mark, Fill, Then Clip Exactly", fontsize=16, fontweight="bold")

    _draw_demo_grid(
        axes[0],
        polygon_gdf,
        x_edges_demo,
        y_edges_demo,
        boundary_mask,
        np.zeros_like(interior_mask),
        "1. Boundary cells",
        "Ring segments mark every crossed cell\nfor exact processing.",
    )
    _draw_demo_grid(
        axes[1],
        polygon_gdf,
        x_edges_demo,
        y_edges_demo,
        boundary_mask,
        interior_mask,
        "2. Interior scanline fill",
        "Whole interior cells are written directly\nbetween intersection pairs.",
        scanline_y=y_demo[scanline_index],
        scanline_intersections=scanline_intersections,
        boundary_alpha=0.18,
    )
    _draw_demo_grid(
        axes[2],
        polygon_gdf,
        x_edges_demo,
        y_edges_demo,
        exact_boundary_mask,
        interior_mask,
        "3. Exact boundary pass",
        "Boundary cells keep exact fractions while\ninterior fills stay bulk-written.",
    )

    legend_handles = [
        patches.Patch(color="#fb923c", alpha=0.85, label="boundary cells clipped exactly"),
        patches.Patch(color="#60a5fa", alpha=0.85, label="whole interior cells filled directly"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(f"{save_path}/algorithm_hybrid_steps.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# --- Generate Lines ---
np.random.seed(0)
lines = []
for _ in range(NUM_LINES):
    start = (
        np.random.rand() * (GRID_X_MAX - GRID_X_MIN) + GRID_X_MIN,
        np.random.rand() * (GRID_Y_MAX - GRID_Y_MIN) + GRID_Y_MIN,
    )
    end = (
        np.random.rand() * (GRID_X_MAX - GRID_X_MIN) + GRID_X_MIN,
        np.random.rand() * (GRID_Y_MAX - GRID_Y_MIN) + GRID_Y_MIN,
    )
    lines.append(LineString([start, end]))

lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)

# Rasterize lines (binary)
lines_binary = rasterize_lines(lines_gdf, x_centers, y_centers, crs, mode="binary")
plt.figure()
plt.pcolormesh(x_edges, y_edges, lines_binary.values, cmap="jet")
lines_gdf.plot(ax=plt.gca(), ec="k", lw=0.7)
plt.gca().set_aspect("equal", adjustable="box")
plt.gca().axis("off")
plt.title("Lines - Binary")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/lines_binary.png", bbox_inches="tight")

# Rasterize lines (length)
lines_length = rasterize_lines(lines_gdf, x_centers, y_centers, crs, mode="length")
plt.figure()
plt.pcolormesh(x_edges, y_edges, lines_length.values, cmap="jet")
lines_gdf.plot(ax=plt.gca(), ec="k", lw=0.7)
plt.colorbar(shrink=0.6, label="Length")
plt.gca().set_aspect("equal", adjustable="box")
plt.gca().axis("off")
plt.title("Lines - Length")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/lines_length.png", bbox_inches="tight")


# --- Generate Polygons ---
np.random.seed(8)
polygons = []
for _ in range(NUM_POLYGONS):
    point = Point(
        np.random.rand() * (GRID_X_MAX - GRID_X_MIN) + GRID_X_MIN,
        np.random.rand() * (GRID_Y_MAX - GRID_Y_MIN) + GRID_Y_MIN,
    )
    polygons.append(point.buffer(0.5))

polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
polygons_gdf = gpd.clip(
    polygons_gdf,
    mask=(GRID_X_MIN, GRID_Y_MIN, GRID_X_MAX, GRID_Y_MAX),
)

# Rasterize polygons (binary)
polygons_binary = rasterize_polygons(polygons_gdf, x_centers, y_centers, crs, mode="binary")
plt.figure()
plt.pcolormesh(x_edges, y_edges, polygons_binary.values, cmap="jet")
polygons_gdf.plot(ax=plt.gca(), fc="none", ec="k", lw=0.7)
plt.gca().set_aspect("equal", adjustable="box")
plt.gca().axis("off")
plt.title("Polygons - Binary")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/polygons_binary.png", bbox_inches="tight")

# Rasterize polygons (area)
polygons_area = rasterize_polygons(polygons_gdf, x_centers, y_centers, crs, mode="area")
plt.figure()
plt.pcolormesh(x_edges, y_edges, polygons_area.values, cmap="jet")
polygons_gdf.plot(ax=plt.gca(), fc="none", ec="k", lw=0.7)
plt.colorbar(shrink=0.6, label="Area")
plt.gca().set_aspect("equal", adjustable="box")
plt.gca().axis("off")
plt.title("Polygons - Area")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/polygons_area.png", bbox_inches="tight")

generate_algorithm_hybrid_figure(SAVE_PATH)
