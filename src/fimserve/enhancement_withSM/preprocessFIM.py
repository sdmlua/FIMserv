"""
Author: Supath Dhital
Date: Jan, 2026

This module contains functions to preprocess the FIM outputs and othe forcings for Surrogate Model based enhancement.
"""

import os
import rasterio
import fiona
from typing import Union, List, Dict, Any
from rasterio.crs import CRS
import numpy as np
import shutil
from pathlib import Path
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_geom
from rasterio.features import bounds as geom_bounds
import warnings

from .utlis import *
from .interactS3 import *

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

    # Convert to binary
    binary_image = (out_image > 0).astype("uint8")

    # Save the binary raster
    with rasterio.open(final_raster_path, "w", **out_meta) as dst:
        dst.write(binary_image)


#Masking with PWB and save the final raster
def mask_with_PWB(input_raster_path, output_raster_path, input_depth=None, output_depth=None):
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

# Clip forcings by a boundary is user is providing the boundary that falls within preparing HUC8
def _bbox_overlaps(b1, b2) -> bool:
    """
    Basic bbox overlap test:
      b = (minx, miny, maxx, maxy)
    """
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def _load_boundary_geometries_from_vector(vector_path: Union[str, Path]):
    """
    Read geometries and CRS from a vector file (gpkg/shp/geojson).
    Returns:
      shapes: list of geometry dicts
      crs: CRS (rasterio CRS) or None
    """
    vector_path = str(vector_path)
    with fiona.open(vector_path, "r") as src:
        shapes = [feat["geometry"] for feat in src if feat and feat.get("geometry")]
        # Fiona may provide CRS as WKT or mapping. Handle robustly.
        fiona_crs = src.crs_wkt or src.crs
    crs = CRS.from_user_input(fiona_crs) if fiona_crs else None
    return shapes, crs


def _ensure_list_of_geoms_and_crs(boundary_geometry, boundary_crs: Union[str, dict] = "EPSG:4326"):
    """
    Normalize boundary input to:
      geoms: list of GeoJSON-like geometry dicts (for rasterio.mask.mask)
      crs:   rasterio CRS describing those geometries

    Supports:
    - geometry dict
    - list/tuple of geometry dicts
    - vector path string (.gpkg/.shp/.geojson/.json)
    """
    if boundary_geometry is None:
        return [], None

    # If user passed a path, load shapes + CRS from file
    if isinstance(boundary_geometry, (str, Path)):
        p = Path(boundary_geometry)
        if p.exists() and p.is_file():
            geoms, crs_from_file = _load_boundary_geometries_from_vector(p)
            if not geoms:
                raise ValueError(f"No geometries found in boundary file: {p}")
            # If file has no CRS, fall back to provided boundary_crs
            crs = crs_from_file if crs_from_file is not None else CRS.from_user_input(boundary_crs)
            return geoms, crs

        # If it is a string but not a file, treat as invalid
        raise ValueError(f"clip_boundary was a string but not a valid file path: {boundary_geometry}")

    # If user passed list of shapes
    if isinstance(boundary_geometry, (list, tuple)):
        if len(boundary_geometry) == 0:
            return [], CRS.from_user_input(boundary_crs)
        return list(boundary_geometry), CRS.from_user_input(boundary_crs)

    # Otherwise assume single geometry dict
    return [boundary_geometry], CRS.from_user_input(boundary_crs)


def _union_bounds(geoms: List[Dict[str, Any]]):
    """
    Build a single bbox from many geometries.
    """
    b0 = geom_bounds(geoms[0])
    minx, miny, maxx, maxy = b0
    for g in geoms[1:]:
        b = geom_bounds(g)
        minx = min(minx, b[0])
        miny = min(miny, b[1])
        maxx = max(maxx, b[2])
        maxy = max(maxy, b[3])
    return (minx, miny, maxx, maxy)


def clip_raster_inplace_to_boundary(
    raster_path: Path,
    boundary_geometry,
    boundary_crs: Union[str, dict] = "EPSG:4326",
    nodata_value: int = 0
) -> Path:
    """
    Clip a raster to boundary_geometry, write to <stem>_Clipped.tif,
    delete the original raster, and return the new path.

    Key behavior:
    - Reads the CRS of the forcing raster.
    - Transforms boundary_geometry to the forcing CRS BEFORE clipping.
    - Uses rasterio.mask.mask with filled=True and nodata=0 to ensure nothing
      persists outside the boundary (no "black dots"/artifacts outside the clip).
    """
    raster_path = Path(raster_path)
    clipped_path = raster_path.with_name(f"{raster_path.stem}_Clipped{raster_path.suffix}")

    geoms, b_crs = _ensure_list_of_geoms_and_crs(boundary_geometry, boundary_crs=boundary_crs)
    if not geoms:
        raise ValueError("boundary_geometry is empty; cannot clip.")

    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        if b_crs is None:
            b_crs = src_crs

        # Transform boundary geometry into forcing CRS if needed
        if src_crs != b_crs:
            geoms_in_src = [transform_geom(b_crs, src_crs, g, precision=6) for g in geoms]
        else:
            geoms_in_src = geoms

        # Robust clip: filled=True ensures outside boundary becomes nodata_value
        out_image, out_transform = mask(
            src,
            geoms_in_src,
            crop=True,
            filled=True,
            nodata=nodata_value,
            all_touched=False
        )

        # Extra safety: ensure masked-out pixels are exactly nodata_value across all bands
        if src.count == 1:
            out_image = np.where(out_image == out_image, out_image, nodata_value)  # no-op for safety
        # mask() already filled; the key is to force dtype + nodata consistency
        if out_image.dtype != src.dtypes[0]:
            out_image = out_image.astype(src.dtypes[0], copy=False)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src_crs,
            "nodata": nodata_value,
        })

    # Write clipped raster first
    with rasterio.open(clipped_path, "w", **out_meta) as dst:
        dst.write(out_image)

    # Preserve your compression convention
    try:
        compress_tif_lzw(clipped_path)
    except Exception:
        pass

    # Delete older file
    try:
        raster_path.unlink()
    except Exception:
        pass

    return clipped_path


def clip_all_forcings_if_boundary_overlaps(
    forcing_dir: Path,
    boundary_geometry,
    boundary_crs: Union[str, dict] = "EPSG:4326"
) -> Dict[Path, Path]:
    """
    Updated behavior (per request):
    - For each forcing raster, read its CRS and transform the boundary to that CRS for overlap check.
    - If ANY forcing does not overlap the boundary:
        * raise a WARNING (not an exception),
        * DO NOT CLIP ANYTHING,
        * DO NOT DELETE ANY FILES,
        * continue using original forcings.
    - If ALL forcings overlap:
        * clip each forcing raster,
        * save as *_Clipped.tif,
        * delete older files (originals).

    Returns:
      mapping {old_path: new_clipped_path} if clipping happened,
      empty dict if clipping was skipped due to non-overlap.
    """
    forcing_dir = Path(forcing_dir)
    tifs = sorted(forcing_dir.glob("*.tif"))

    if not tifs:
        raise FileNotFoundError(f"No .tif forcing rasters found in: {forcing_dir}")

    geoms, b_crs = _ensure_list_of_geoms_and_crs(boundary_geometry, boundary_crs=boundary_crs)
    if not geoms:
        raise ValueError("boundary_geometry is empty; cannot clip forcings.")

    # Check bbox overlap for ALL rasters first
    non_overlapping = []

    for tif in tifs:
        with rasterio.open(tif) as src:
            if src.crs is None:
                non_overlapping.append(tif)
                continue

            # Transform boundary geometries into this forcing CRS before overlap check
            if b_crs is None:
                b_crs_local = src.crs
            else:
                b_crs_local = b_crs

            if src.crs != b_crs_local:
                geoms_in_src = [transform_geom(b_crs_local, src.crs, g, precision=6) for g in geoms]
            else:
                geoms_in_src = geoms

            boundary_bbox = _union_bounds(geoms_in_src)
            raster_bbox = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

            if not _bbox_overlaps(boundary_bbox, raster_bbox):
                non_overlapping.append(tif)

    # If any do not overlap, warn and skip clipping entirely
    if non_overlapping:
        missing = "\n".join([f" - {p.name}" for p in non_overlapping])
        warnings.warn(
            "Boundary does not overlap with all forcing rasters. "
            "Skipping forcing clipping and keeping original forcing rasters.\n"
            f"Non-overlapping rasters:\n{missing}",
            category=UserWarning
        )
        return {}

    # Clip all rasters
    mapping = {}
    for tif in tifs:
        new_path = clip_raster_inplace_to_boundary(
            raster_path=tif,
            boundary_geometry=boundary_geometry,
            boundary_crs=boundary_crs,
            nodata_value=0
        )
        mapping[tif] = new_path

    return mapping

# PREPROCESS THE OWP HAND BASED FIM FOR SM
def prepare_FORCINGs(
    huc_id,
    event_date=None,
    data='retrospective',
    forecast_range=None,
    forecast_date=None,
    sort_by=None,
    clip_boundary=None,
    clip_boundary_crs: Union[str, dict] = "EPSG:4326"
):

    # GET FORCINGS
    print("Downloading forcings from the S3 bucket...\n")
    get_forcings(huc_id)
    print("Forcings downloaded successfully.\n")

    # If here, some boundary is passed, If that boundary overlaps with all the forcings,
    forcing_dir = Path(f'./HUC{huc_id}_forcings')
    forcing_suffix = ""  

    if clip_boundary is not None:
        print("Boundary provided. Checking overlap with all forcings...\n")
        mapping = clip_all_forcings_if_boundary_overlaps(
            forcing_dir=forcing_dir,
            boundary_geometry=clip_boundary,
            boundary_crs=clip_boundary_crs
        )
        if mapping:
            forcing_suffix = "_clipped"
            print("All forcing rasters clipped successfully.\n")
        else:
            print("Skipping forcing clipping due to non-overlap.\n")

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
        reference_dir = forcing_dir / f'LULC_HUC{huc_id}{forcing_suffix}.tif'
        FIM_finaldir = forcing_dir / f'hand_{FIM.stem}.tif'
        align_raster(final_raster, reference_dir, FIM_finaldir)

    # Clean up temporary FIM directory
    if cwd.exists() and cwd.is_dir():
        shutil.rmtree(cwd)

    print("FIM file preprocessed successfully.\n")
