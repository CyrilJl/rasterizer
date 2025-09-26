import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point

from rasterizer.lines import rasterize_lines
from rasterizer.polygons import rasterize_polygons

# Grid setup
x_centers = np.arange(0.5, 5.5, 1.0)
y_centers = np.arange(0.5, 5.5, 1.0)
dx = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 1.0
dy = y_centers[1] - y_centers[0] if len(y_centers) > 1 else 1.0
x_edges = np.arange(x_centers[0] - dx / 2, x_centers[-1] + dx, dx)
y_edges = np.arange(y_centers[0] - dy / 2, y_centers[-1] + dy, dy)
crs = "EPSG:3857"

# --- Generate Lines ---
np.random.seed(0)
lines = []
for _ in range(8):
    start = (np.random.rand() * 5, np.random.rand() * 5)
    end = (np.random.rand() * 5, np.random.rand() * 5)
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
plt.savefig("docs/_static/lines_binary.png", bbox_inches="tight")

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
plt.savefig("docs/_static/lines_length.png", bbox_inches="tight")


# --- Generate Polygons ---
np.random.seed(1)
polygons = []
for _ in range(5):
    point = Point(np.random.rand() * 5, np.random.rand() * 5)
    polygons.append(point.buffer(0.5))

polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
polygons_gdf = gpd.clip(
    polygons_gdf,
    mask=(
        0,
        0,
        6,
        6,
    ),
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
plt.savefig("docs/_static/polygons_binary.png", bbox_inches="tight")

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
plt.savefig("docs/_static/polygons_area.png", bbox_inches="tight")
