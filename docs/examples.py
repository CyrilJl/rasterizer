import os
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import wget
from mapflow import plot_da

import rasterizer


def download_and_unzip(url, target_dir):
    """Downloads and unzips a file."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    zip_path = os.path.join(target_dir, os.path.basename(url))

    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        wget.download(url, out=zip_path)

    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def main():
    # --- Data URLs ---
    roads_url = "https://maps.dft.gov.uk/major-road-network-shapefile/Major_Road_Network_2018_Open_Roads.zip"
    land_use_url = "https://download.data.public.lu/resources/corine-land-cover-2018/20200325-152907/corine-land-cover-2018-1872-classes-0.shp.zip"

    # --- Download and prepare data ---
    data_dir = "data"
    roads_dir = os.path.join(data_dir, "roads")
    land_use_dir = os.path.join(data_dir, "land_use")

    download_and_unzip(roads_url, roads_dir)
    download_and_unzip(land_use_url, land_use_dir)

    roads_shapefile = os.path.join(roads_dir, "Major_Road_Network_2018_Open_Roads.shp")
    land_use_shapefile = os.path.join(land_use_dir, "Corine_Land_Cover_2018_1872_Classes_0.shp")

    # Create _static directory if it doesn't exist
    if not os.path.exists("docs/_static"):
        os.makedirs("docs/_static")

    resolution = 1000.0  # in meters

    # Rasterize and plot roads
    roads = gpd.read_file(roads_shapefile)
    x_roads = np.arange(roads.total_bounds[0], roads.total_bounds[2], resolution)
    y_roads = np.arange(roads.total_bounds[1], roads.total_bounds[3], resolution)
    raster_roads = rasterizer.rasterize_lines(roads, x_roads, y_roads, crs=roads.crs)

    plot_da(raster_roads, title=f"Roads - {resolution}x{resolution} raster", figsize=(10, 10), log=True, show=False)
    plt.tight_layout()
    plt.savefig("docs/_static/roads_raster.png", bbox_inches="tight")
    plt.close()

    # Rasterize and plot land use
    land_use = gpd.read_file(land_use_shapefile)
    x_lu = np.arange(land_use.total_bounds[0], land_use.total_bounds[2], resolution)
    y_lu = np.arange(land_use.total_bounds[1], land_use.total_bounds[3], resolution)
    raster_lu = rasterizer.rasterize_polygons(land_use, x_lu, y_lu, crs=land_use.crs)

    plot_da(raster_lu, title=f"Land Use - {resolution}x{resolution} raster", figsize=(10, 10), log=True, show=False)
    plt.tight_layout()
    plt.savefig("docs/_static/roads_raster.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
