import os
import s3fs
import tempfile
from pathlib import Path
import geopandas as gpd
from io import BytesIO
import fsspec

# Initialize anonymous S3 filesystem
fs = s3fs.S3FileSystem(anon=True)


# FINDING THE INTERSECTED HUC8
def find_intersecting_huc8ID(huc8_gdf, user_boundary_path):
    user_gdf = gpd.read_file(user_boundary_path)
    if user_gdf.crs != "EPSG:4326":
        user_gdf = user_gdf.to_crs("EPSG:4326")
    intersecting = gpd.overlay(huc8_gdf, user_gdf, how="intersection")

    if intersecting.empty:
        return "No HUC8 region intersects with the user-defined boundary."

    output = ["Your boundary falls within:\n----------------------------------"]
    for _, row in intersecting.iterrows():
        output.append(f"HUC8 - {row['HUC8']}\nNAME - {row['name']}\n")
    return "\n".join(output)


# GETTING THE HUC8 BOUNDARIES FROM S3
def HUC8_inS3(fs, bucket, prefix="HUC8_boundaries/"):
    # List available files
    files = fs.ls(f"{bucket}/{prefix}")
    gpkg_key = next((f for f in files if f.endswith(".gpkg")), None)

    if gpkg_key is None:
        raise FileNotFoundError(f"No .gpkg file found in s3://{bucket}/{prefix}")

    # Read file into temporary file
    with fs.open(gpkg_key, "rb") as s3file:
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            tmp_file.write(s3file.read())
            tmp_path = tmp_file.name

    gdf = gpd.read_file(tmp_path)
    return gdf


# WRAPPING ALL FUNCTIONS
def getIntersectedHUC8ID(user_boundary):
    bucket_name = "sdmlab"
    HUC8_gdf = HUC8_inS3(fs, bucket_name)
    HUC8 = find_intersecting_huc8ID(HUC8_gdf, user_boundary)
    return HUC8
