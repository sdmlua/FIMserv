import os
import glob
import rasterio
from rasterio.mask import mask
from pathlib import Path
import geopandas as gpd

from ..datadownload import setup_directories


def checkSHP(input_file):
    if isinstance(input_file, Path):
        input_file = str(input_file)

    if input_file.endswith(".shp"):
        gdf = gpd.read_file(input_file)
    elif input_file.endswith(".gpkg"):
        gdf = gpd.read_file(input_file)
    elif input_file.endswith(".geojson"):
        gdf = gpd.read_file(input_file)
    elif input_file.endswith(".kml"):
        gdf = gpd.read_file(input_file)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a shapefile GeoPackage, GeoJSON, or KML."
        )
    return gdf


def clipFIMforboundary(inundation_raster, shapefile_geom, output_directory, huc):
    with rasterio.open(inundation_raster) as src:
        geometries = shapefile_geom.geometry.tolist()
        if src.crs != shapefile_geom.crs:
            print("CRS mismatch. Reprojecting geometry to match raster.")
            geometries = [geom.to_crs(src.crs) for geom in geometries]

        out_image, out_transform = mask(src, geometries, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "count": 1,
                "crs": src.crs,
                "transform": out_transform,
                "width": out_image.shape[2],
                "height": out_image.shape[1],
            }
        )

        output_directory = os.path.join(output_directory, "subsetFIM")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        file_basename = os.path.basename(inundation_raster).split("_")[0]
        output_file = os.path.join(output_directory, f"{file_basename}_subsetFIM.tif")

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)
        print(f"Clipped raster saved to {output_file}")
