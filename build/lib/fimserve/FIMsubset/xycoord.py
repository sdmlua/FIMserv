import os
import glob
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS
import rasterio
from rasterio.mask import mask

from ..datadownload import setup_directories
from .shpsubset import checkSHP, clipFIMforboundary


def checkifWGS(x, y):
    """Check if coordinates are in WGS84 (EPSG:4326) range."""
    if -180 <= x <= 180 and -90 <= y <= 90:
        return True
    else:
        return False


def reproject_coordinates(x, y, target_crs="EPSG:5070"):
    """Reproject coordinates to target CRS."""
    if checkifWGS(x, y):
        from pyproj import CRS, Transformer

        crs_wgs84 = CRS("EPSG:4326")
        crs_target = CRS(target_crs)
        transformer = Transformer.from_crs(crs_wgs84, crs_target, always_xy=True)
        x_new, y_new = transformer.transform(x, y)
        return x_new, y_new
    else:
        return x, y


def clipFIM(inundation_raster, shapefile_geom, output_directory, huc):
    with rasterio.open(inundation_raster) as src:
        out_image, out_transform = mask(src, [shapefile_geom], crop=True)
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
        file_basename = os.path.basename(inundation_raster).split("_")[0]
        output_file = os.path.join(output_directory, f"{file_basename}_subsetFIM.tif")

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Clipped raster saved to {output_file}")
        return out_image, out_transform


def withininWatershed(x, y, geopackage_path, inundation_raster, huc, output_directory):
    point = Point(x, y)
    watersheds = gpd.read_file(geopackage_path)

    # Watershed containing the point
    watershed_containing_point = None
    for _, watershed in watersheds.iterrows():
        if watershed["geometry"].contains(point):
            watershed_containing_point = watershed
            break
    output_directory = os.path.join(output_directory, "subsetFIM")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if watershed_containing_point is None:
        raise ValueError(f"Point is not within any watershed boundary for {huc}")
    return clipFIM(
        inundation_raster, watershed_containing_point["geometry"], output_directory, huc
    )


def subsetFIM(location, huc, method):
    code_dir, data_dir, output_dir = setup_directories()
    gpkg_path = os.path.join(
        output_dir, f"flood_{huc}", huc, "nwm_catchments_proj_subset.gpkg"
    )
    out_dir = os.path.join(output_dir, f"flood_{huc}", f"{huc}_inundation")
    inundation_rasters = os.path.join(out_dir, "*_inundation.tif")
    print(inundation_rasters)
    files = glob.glob(inundation_rasters)
    if method == "xy":
        x, y = location
        print("point", x, y)
        x, y = reproject_coordinates(x, y)
        for file in files:
            print(f"Clipping {file} to watershed containing point {x}, {y}")
            withininWatershed(x, y, gpkg_path, file, huc, out_dir)
    elif method == "boundary":
        shapefile = checkSHP(location)
        for file in files:
            clipFIMforboundary(file, shapefile, out_dir, huc)
