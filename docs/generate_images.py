import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from rasterizer.lines import rasterize_lines
from rasterizer.polygons import rasterize_polygons

# Grid setup
x = np.arange(0.5, 5.5, 1.0)
y = np.arange(0.5, 5.5, 1.0)
crs = "EPSG:3857"

# --- Generate Lines ---
lines = []
for _ in range(3):
    start = (np.random.rand() * 5, np.random.rand() * 5)
    end = (np.random.rand() * 5, np.random.rand() * 5)
    lines.append(LineString([start, end]))

lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)

# Rasterize lines (binary)
lines_binary = rasterize_lines(lines_gdf, x, y, crs, mode='binary')
plt.figure()
plt.imshow(lines_binary, cmap='jet', origin='lower')
lines_gdf.plot(ax=plt.gca(), ec='k', lw=0.7)
plt.gca().axis('off')
plt.title("Lines - Binary")
plt.savefig("docs/lines_binary.png")

# Rasterize lines (length)
lines_length = rasterize_lines(lines_gdf, x, y, crs, mode='length')
plt.figure()
plt.imshow(lines_length, cmap='jet', origin='lower')
lines_gdf.plot(ax=plt.gca(), ec='k', lw=0.7)
plt.colorbar(shrink=0.6, label="Length")
plt.gca().axis('off')
plt.title("Lines - Length")
plt.savefig("docs/lines_length.png")


# --- Generate Polygons ---
polygons = []
for _ in range(3):
    coords = [(np.random.rand() * 5, np.random.rand() * 5) for _ in range(4)]
    polygons.append(Polygon(coords))

polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

# Rasterize polygons (binary)
polygons_binary = rasterize_polygons(polygons_gdf, x, y, crs, mode='binary')
plt.figure()
plt.imshow(polygons_binary, cmap='jet', origin='lower')
polygons_gdf.plot(ax=plt.gca(), fc='none', ec='k', lw=0.7)
plt.gca().axis('off')
plt.title("Polygons - Binary")
plt.savefig("docs/polygons_binary.png")

# Rasterize polygons (area)
polygons_area = rasterize_polygons(polygons_gdf, x, y, crs, mode='area')
plt.figure()
plt.imshow(polygons_area, cmap='jet', origin='lower')
polygons_gdf.plot(ax=plt.gca(), fc='none', ec='k', lw=0.7)
plt.colorbar(shrink=0.6, label="Area")
plt.gca().axis('off')
plt.title("Polygons - Area")
plt.savefig("docs/polygons_area.png")

print("Generated images: lines_binary.png, lines_length.png, polygons_binary.png, polygons_area.png")