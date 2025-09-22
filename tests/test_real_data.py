import geopandas as gpd
import numpy as np
import rasterizer
import os
import wget
import zipfile
import pytest

def download_and_unzip(url, target_dir):
    """Downloads and unzips a file."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    zip_path = os.path.join(target_dir, os.path.basename(url))

    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        wget.download(url, out=zip_path)

    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

@pytest.fixture(scope="session")
def roads_data():
    roads_url = "https://maps.dft.gov.uk/major-road-network-shapefile/Major_Road_Network_2018_Open_Roads.zip"
    data_dir = "data"
    roads_dir = os.path.join(data_dir, "roads")
    download_and_unzip(roads_url, roads_dir)
    shapefile_path = os.path.join(roads_dir, "Major_Road_Network_2018_Open_Roads.shp")
    return gpd.read_file(shapefile_path)

@pytest.fixture(scope="session")
def land_use_data():
    land_use_url = "https://download.data.public.lu/resources/corine-land-cover-2018/20200325-152907/corine-land-cover-2018-1872-classes-0.shp.zip"
    data_dir = "data"
    land_use_dir = os.path.join(data_dir, "land_use")
    download_and_unzip(land_use_url, land_use_dir)
    shapefile_path = os.path.join(land_use_dir, "Corine_Land_Cover_2018_1872_Classes_0.shp")
    return gpd.read_file(shapefile_path)

def test_rasterize_real_lines(roads_data):
    x = np.linspace(roads_data.total_bounds[0], roads_data.total_bounds[2], 100)
    y = np.linspace(roads_data.total_bounds[1], roads_data.total_bounds[3], 100)
    raster = rasterizer.rasterize_lines(roads_data, x, y, crs=roads_data.crs)

    assert raster.shape == (100, 100)
    assert raster.dtype == np.float32
    assert raster.sum() > 0

def test_rasterize_real_polygons(land_use_data):
    x = np.linspace(land_use_data.total_bounds[0], land_use_data.total_bounds[2], 100)
    y = np.linspace(land_use_data.total_bounds[1], land_use_data.total_bounds[3], 100)
    raster = rasterizer.rasterize_polygons(land_use_data, x, y, crs=land_use_data.crs)

    assert raster.shape == (100, 100)
    assert raster.dtype == np.float32
    assert raster.sum() > 0
