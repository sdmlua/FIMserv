import s3fs
import geopandas as gpd
import os
import tempfile
import fiona
import rasterio
from pathlib import Path
import numpy as np
from rasterio.mask import mask

fs = s3fs.S3FileSystem(anon=True)
bucket_name = "sdmlab"

# FINDING THE INTERSECTED HUC8 AND RETURNING GEOMETRY IN WGS84
def HUC8_inS3(fs, bucket, prefix="HUC8_boundaries/"):
    files = fs.ls(f"{bucket}/{prefix}")
    gpkg_key = next((f for f in files if f.endswith('.gpkg')), None)

    if gpkg_key is None:
        raise FileNotFoundError(f"No .gpkg file found in s3://{bucket}/{prefix}")

    # Download and read GeoPackage into a GeoDataFrame
    with fs.open(gpkg_key, 'rb') as s3file:
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            tmp_file.write(s3file.read())
            tmp_path = tmp_file.name
    
    gdf = gpd.read_file(tmp_path)
    return gdf

# WRAPPING ALL FUNCTIONS
def getHUC8BoundaryByID(huc_id):
    huc8_gdf = HUC8_inS3(fs, bucket_name)
    if huc8_gdf.crs != "EPSG:4326":
        huc8_gdf = huc8_gdf.to_crs("EPSG:4326")
    selected = huc8_gdf[huc8_gdf['HUC8'] == huc_id]
    if selected.empty:
        raise ValueError(f"HUC8 ID {huc_id} not found in the boundary file.")
    
    return selected.geometry


#PWB of CONUS rivers
def PWB_inS3(fs, bucket, prefix="PWB/"):
    tmp_dir = tempfile.mkdtemp()
    files = fs.ls(f"{bucket}/{prefix}")

    # Filter out relevant shapefile components
    for file_key in files:
        file_name = os.path.basename(file_key)
        if file_name.endswith(('.shp', '.shx', '.dbf', '.prj', '.cpg')):
            with fs.open(file_key, 'rb') as s3file:
                local_path = os.path.join(tmp_dir, file_name)
                with open(local_path, 'wb') as local_file:
                    local_file.write(s3file.read())

    # Ensure we got a .shp file
    shp_files = [f for f in os.listdir(tmp_dir) if f.endswith(".shp")]
    if not shp_files:
        raise ValueError("No .shp file found after download.")

    return os.path.join(tmp_dir, shp_files[0])

#GET FORCINGS
def get_forcings(huc_id, downloadforcings=True):
    s3_prefix = f"SM_dataset/HUCIDs_forcings/HUC{huc_id}/"  
    local_folder = Path(f"HUC{huc_id}_forcings")
    local_folder.mkdir(exist_ok=True)

    if downloadforcings:
        fs = s3fs.S3FileSystem(anon=True)
        bucket_name = "sdmlab"
        s3_files = fs.ls(f"{bucket_name}/{s3_prefix}")

        if not s3_files:
            raise FileNotFoundError(f"No files found at s3://{bucket_name}/{s3_prefix}")

        # Download all files
        for s3_path in s3_files:
            relative_path = Path(s3_path.replace(f"{bucket_name}/{s3_prefix}", ""))
            local_path = local_folder / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get(s3_path, str(local_path))
    else:
        print(f"Skipping download; using existing benchmark data for HUC{huc_id}")

def get_population_GRID(boundary_gdf, fs = fs, bucket=bucket_name, prefix="SM_dataset/gridded_population/"):
    files = fs.ls(f"{bucket}/{prefix}")
    tif_key = next((f for f in files if f.endswith('.tif')), None)

    if tif_key is None:
        raise FileNotFoundError(f"No .tif file found in s3://{bucket}/{prefix}")

    with fs.open(tif_key, 'rb') as s3file:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_file.write(s3file.read())
            tmp_tif_path = tmp_file.name

    # Clip gridded population raster with geometry
    with rasterio.open(tmp_tif_path) as src:
        geoms = [geom.__geo_interface__ for geom in boundary_gdf.geometry]
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    return out_image, out_meta
