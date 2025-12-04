import os
import rasterio
import geopandas as gpd
import fiona
from typing import Union
from rasterio.crs import CRS
import numpy as np
import shutil
from pathlib import Path
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

from utlis import *
from interactS3 import *

#Import the Streamflow data Download and FIM running module
from ..datadownload import DownloadHUC8
from ..streamflowdata.nwmretrospective import getNWMretrospectivedata
from ..streamflowdata.forecasteddata import getNWMForecasteddata
from ..runFIM import runOWPHANDFIM

#GET LOW FIDELITY USING FIMSERVE
def get_LFFIM(huc_id, event_date=None, data='forecast', forecast_range=None, forecast_date=None, sort_by=None):
    original_cwd = os.getcwd()
    try:
        createCWD('fim')
        DownloadHUC8(huc_id)
        
        # For retrospective event
        if data == 'retrospective':
            if not event_date:
                raise ValueError("event_date is required for retrospective analysis.")
            huc_event_dict = initialize_huc_event(huc_id, event_date)
            getNWMretrospectivedata(huc_event_dict=huc_event_dict)
        
        # For forecasting event
        elif data == 'forecast':
            if not forecast_range:
                raise ValueError("forecast_range ('short_range', 'medium_range', or 'long_range') is required for forecast.")
            
            if forecast_range in ['medium_range', 'long_range']:
                if not sort_by:
                    sort_by = 'maximum'
                getNWMForecasteddata(
                    huc_id=huc_id,
                    forecast_range=forecast_range,
                    forecast_date=forecast_date,
                    sort_by=sort_by
                )
            else:
                getNWMForecasteddata(
                    huc_id=huc_id,
                    forecast_range=forecast_range,
                    forecast_date=forecast_date
                )
        else:
            raise ValueError("data_type must be either 'retrospective' or 'forecast'.")
        
        # Run the FIM
        runOWPHANDFIM(huc_id)

    finally:
        os.chdir(original_cwd)  


def load_shapes(shapefile_path):
    with fiona.open(shapefile_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    return shapes

# Remove permanent water bodies from the raster data
def remove_water_bodies(raster_path, PWB_water):
    with fiona.open(PWB_water, "r") as shapefile:
        water_bodies_shapes = [feature["geometry"] for feature in shapefile]

    # Read the masked raster file
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, water_bodies_shapes, invert=True)
        out_image = np.where((out_image != 0) & (out_image != 1), 0, out_image)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
    return out_image, out_meta

#Reproject raster if needed
def reproject_raster(
    input_raster_path: str,
    output_file: str,
    target_crs: Union[str, dict] = "EPSG:4326",
    target_resolution: float = 8.983152841195214829e-05
):
    if isinstance(target_crs, dict):
        target_crs = CRS.from_user_input(target_crs)

    # Read and reproject raster
    with rasterio.open(input_raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution,
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "driver": "GTiff",
        })

        reprojected_data = np.empty((src.count, height, width), dtype=src.dtypes[0])

        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=reprojected_data[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
            )

    # Save reprojected raster
    with rasterio.open(output_file, "w", **kwargs) as dst:
        dst.write(reprojected_data.squeeze(), 1)


#Raster to binary
def raster2binary(
    input_raster_path,
    geometry, 
    final_raster_path
):
    # Mask the raster with the geometry
    with rasterio.open(input_raster_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True, filled=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs,
            "nodata": 0,
        })

    # Convert to binary (HAND logic: flooded if value > 0)
    binary_image = (out_image > 0).astype("uint8")

    # Save the binary raster
    with rasterio.open(final_raster_path, "w", **out_meta) as dst:
        dst.write(binary_image)

#Masking with PWB and save the final raster
def mask_with_PWB(input_raster_path, output_raster_path, input_depth = None, output_depth = None):
    PWB_shp = PWB_inS3(fs, bucket_name)
    shapes = load_shapes(PWB_shp)

    with rasterio.open(input_raster_path) as src:
        out_image, out_transform = mask(src, shapes, invert=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs
        })

    with rasterio.open(output_raster_path, "w", **out_meta) as dst:
        dst.write(out_image)

    if input_depth and output_depth:
        with rasterio.open(input_depth) as src:
            out_image, out_transform = mask(src, shapes, invert=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs
            })

        with rasterio.open(output_depth, "w", **out_meta) as dst:
            dst.write(out_image)
            
#Align the raster to the reference raster
def align_raster(
    hand_fim_raster_path: str,
    reference_raster_path: str,
    output_fim_aligned_path: str
):
    with rasterio.open(reference_raster_path) as ref:
        ref_meta = ref.meta.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(hand_fim_raster_path) as src:
        src_data = src.read(1) 
        src_crs = src.crs
        src_transform = src.transform
        src_dtype = src.dtypes[0]
        aligned_data = np.zeros((ref_height, ref_width), dtype=src_dtype)
        reproject(
            source=src_data,
            destination=aligned_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_width=ref_width,
            dst_height=ref_height,
            resampling=Resampling.nearest
        )

    ref_meta.update({
    "driver": "GTiff",
    "dtype": src_dtype,
    "count": 1,
    "compress": "lzw",
    "nodata": 0 
})
    
    with rasterio.open(output_fim_aligned_path, "w", **ref_meta) as dst:
        dst.write(aligned_data, 1)

# PREPROCESS THE OWP HAND BASED FIM FOR SM
def prepare_FORCINGs(huc_id, event_date=None, data='retrospective', forecast_range=None, forecast_date=None, sort_by=None):
    
    # GET FORCINGS
    print("Downloading forcings from the S3 bucket...\n")
    get_forcings(huc_id)
    print("Forcings downloaded successfully.\n")
    
    # GET THE FIM FILES
    print(f"Generating the FIM files for {data} event...\n")
    get_LFFIM(huc_id, event_date=event_date, data=data, forecast_range=forecast_range, forecast_date=forecast_date, sort_by=sort_by)
    print("FIM files generated successfully.\n")
    
    # PREPROCESSING THE FIM FILES
    print("Preprocessing the FIM files...\n")
    cwd = Path('./fim')
    fim_dir = cwd / f'output/flood_{huc_id}/{huc_id}_inundation/'
    fim_files = sorted(fim_dir.glob("*.tif"))
    
    # Get the HUC8 boundary
    HUC_boundary = getHUC8BoundaryByID(huc_id)
        
    for FIM in fim_files:
        out_dir = fim_dir / 'processing'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Reproject the FIM file
        FIM_file = out_dir / f'{FIM.stem}_reprojected.tif'
        reproject_raster(FIM, FIM_file)
        compress_tif_lzw(FIM_file)
        
        # Convert to binary
        out_dir_binary = out_dir / f'{FIM.stem}_binary.tif'
        raster2binary(FIM_file, HUC_boundary, out_dir_binary)
        compress_tif_lzw(out_dir_binary)
        
        # Mask and clip with PWB
        final_raster = out_dir / f'{FIM.stem}_final.tif'
        mask_with_PWB(out_dir_binary, final_raster)
        compress_tif_lzw(final_raster)
        
        # Align final FIM raster with reference raster
        forcing_dir = Path(f'./HUC{huc_id}_forcings')
        reference_dir = forcing_dir / f'LULC_HUC{huc_id}.tif'
        FIM_finaldir = forcing_dir / f'hand_{FIM.stem}.tif'
        align_raster(final_raster, reference_dir, FIM_finaldir)

    # Clean up temporary FIM directory
    if cwd.exists() and cwd.is_dir():
        shutil.rmtree(cwd)

    print("FIM file preprocessed successfully.\n")
