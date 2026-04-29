"""
Author: Supath Dhital
Date updated: Apr, 2026

This module contains functions to preprocess the FIM outputs and othe forcings for Surrogate Model based enhancement.
"""

import os
import pandas as pd
import rasterio
import fiona
from typing import Union, List, Dict, Any
from rasterio.crs import CRS
import numpy as np
import shutil
from pathlib import Path
from rasterio.mask import mask
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_geom,
)
from rasterio.features import bounds as geom_bounds
import warnings
import logging

from .utlis import *
from .interactS3 import *
from fimeval.ContingencyMap.water_bodies import ExtractPWB

# Import the Streamflow data Download and FIM running module
from ..datadownload import DownloadHUC8, setup_directories
from ..streamflowdata.nwmretrospective import getNWMretrospectivedata
from ..streamflowdata.forecasteddata import getNWMForecasteddata
from ..runFIM import runOWPHANDFIM

logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# GET LOW FIDELITY USING FIMSERVE
def get_LFFIM(
    huc_id,
    event_date=None,
    data="forecast",
    forecast_range=None,
    forecast_date=None,
    sort_by=None,
):
    DownloadHUC8(huc_id)

    # For retrospective event
    if data == "retrospective":
        if not event_date:
            raise ValueError("event_date is required for retrospective analysis.")
        huc_event_dict = initialize_huc_event(huc_id, event_date)
        getNWMretrospectivedata(huc_event_dict=huc_event_dict)

    # For forecasting event
    elif data == "forecast":
        if not forecast_range:
            raise ValueError(
                "forecast_range ('short_range', 'medium_range', or 'long_range') is required for forecast."
            )

        if forecast_range in ["medium_range", "long_range"]:
            if not sort_by:
                sort_by = "maximum"
            getNWMForecasteddata(
                huc_id=huc_id,
                forecast_range=forecast_range,
                forecast_date=forecast_date,
                sort_by=sort_by,
            )
        else:
            getNWMForecasteddata(
                huc_id=huc_id,
                forecast_range=forecast_range,
                forecast_date=forecast_date,
            )
    else:
        raise ValueError("data_type must be either 'retrospective' or 'forecast'.")

    # Run the FIM
    runOWPHANDFIM(huc_id)


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


# Reproject raster if needed
def reproject_raster(
    input_raster_path: str,
    output_file: str,
    target_crs: Union[str, dict] = "EPSG:4326",
    target_resolution: float = 8.983152841195214829e-05,
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
        kwargs.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "driver": "GTiff",
            }
        )

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


# Raster to binary
def raster2binary(input_raster_path, geometry, final_raster_path):
    # Mask the raster with the geometry
    with rasterio.open(input_raster_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True, filled=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs,
                "nodata": 0,
            }
        )

    # Convert to binary
    binary_image = (out_image > 0).astype("uint8")

    # Save the binary raster
    with rasterio.open(final_raster_path, "w", **out_meta) as dst:
        dst.write(binary_image)


# Masking with PWB and save the final raster
def mask_with_PWB(
    input_raster_path, output_raster_path, boundary, input_depth=None, output_depth=None
):
    import geopandas as gpd
    if not isinstance(boundary, gpd.GeoDataFrame):
        boundary = gpd.GeoDataFrame(geometry=boundary, crs="EPSG:4326")
    pwb = ExtractPWB(boundary=boundary, save=False, output_filename="permanent_water.gpkg")
    shapes = [geom.__geo_interface__ for geom in pwb.gdf.geometry if geom is not None]

    with rasterio.open(input_raster_path) as src:
        out_image, out_transform = mask(src, shapes, invert=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs,
            }
        )

    with rasterio.open(output_raster_path, "w", **out_meta) as dst:
        dst.write(out_image)

    if input_depth and output_depth:
        with rasterio.open(input_depth) as src:
            out_image, out_transform = mask(src, shapes, invert=True)
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": src.crs,
                }
            )

        with rasterio.open(output_depth, "w", **out_meta) as dst:
            dst.write(out_image)


# Align the raster to the reference raster
def align_raster(
    hand_fim_raster_path: str, reference_raster_path: str, output_fim_aligned_path: str
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
            resampling=Resampling.nearest,
        )

    ref_meta.update(
        {
            "driver": "GTiff",
            "dtype": src_dtype,
            "count": 1,
            "compress": "lzw",
            "nodata": 0,
        }
    )

    with rasterio.open(output_fim_aligned_path, "w", **ref_meta) as dst:
        dst.write(aligned_data, 1)


# Clip forcings by a boundary is user is providing the boundary that falls within preparing HUC8
def _normalize_bbox(b) -> tuple:
    """
    Return (minx, miny, maxx, maxy) regardless of whether the raster has a
    flipped geotransform (negative-north / bottom > top).
    """
    minx, miny, maxx, maxy = b
    if minx > maxx:
        minx, maxx = maxx, minx
    if miny > maxy:
        miny, maxy = maxy, miny
    return (minx, miny, maxx, maxy)


def _bbox_overlaps(b1, b2) -> bool:
    """
    Overlap test for (minx, miny, maxx, maxy) bboxes.
    Normalizes both inputs first so flipped raster bounds never cause false negatives.
    """
    b1 = _normalize_bbox(b1)
    b2 = _normalize_bbox(b2)
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def _load_boundary_geometries_from_vector(vector_path: Union[str, Path]):
    """
    Read geometries and CRS from a vector file (gpkg/shp/geojson).
    Returns:
      shapes: list of geometry dicts
      crs: rasterio CRS or None
    """
    import geopandas as gpd
    gdf = gpd.read_file(str(vector_path))
    if gdf.crs is None:
        return [f.__geo_interface__ for f in gdf.geometry if f is not None], None
    crs = CRS.from_user_input(gdf.crs.to_wkt())
    shapes = [f.__geo_interface__ for f in gdf.geometry if f is not None]
    return shapes, crs


def _ensure_list_of_geoms_and_crs(
    boundary_geometry, boundary_crs: Union[str, dict] = "EPSG:4326"
):
    """
    Normalize boundary input to:
      geoms: list of GeoJSON-like geometry dicts (for rasterio.mask.mask)
      crs:   rasterio CRS describing those geometries

    Supports:
    - file path string/Path (.gpkg/.shp/.geojson/.json) — CRS read from file
    - GeoDataFrame — CRS read from GeoDataFrame
    - geometry dict or list of geometry dicts — uses boundary_crs
    """
    import geopandas as gpd

    if boundary_geometry is None:
        return [], None

    # File path — read via geopandas to reliably get CRS
    if isinstance(boundary_geometry, (str, Path)):
        p = Path(boundary_geometry)
        if not p.exists() or not p.is_file():
            raise ValueError(f"clip_boundary path does not exist: {boundary_geometry}")
        geoms, crs = _load_boundary_geometries_from_vector(p)
        if not geoms:
            raise ValueError(f"No geometries found in boundary file: {p}")
        if crs is None:
            print(f"Warning: boundary file has no CRS, assuming {boundary_crs}")
            crs = CRS.from_user_input(boundary_crs)
        return geoms, crs

    # GeoDataFrame — extract geometries and CRS directly
    if isinstance(boundary_geometry, gpd.GeoDataFrame):
        if boundary_geometry.crs is None:
            print(f"Warning: boundary GeoDataFrame has no CRS, assuming {boundary_crs}")
            crs = CRS.from_user_input(boundary_crs)
        else:
            crs = CRS.from_user_input(boundary_geometry.crs.to_wkt())
        geoms = [f.__geo_interface__ for f in boundary_geometry.geometry if f is not None]
        if not geoms:
            raise ValueError("boundary GeoDataFrame has no valid geometries.")
        return geoms, crs

    # List/tuple of geometry dicts
    if isinstance(boundary_geometry, (list, tuple)):
        if len(boundary_geometry) == 0:
            return [], CRS.from_user_input(boundary_crs)
        return list(boundary_geometry), CRS.from_user_input(boundary_crs)

    # Single geometry dict
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
    nodata_value: int = 0,
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
    clipped_path = raster_path.with_name(
        f"{raster_path.stem}_clipped{raster_path.suffix}"
    )

    geoms, b_crs = _ensure_list_of_geoms_and_crs(
        boundary_geometry, boundary_crs=boundary_crs
    )
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
            geoms_in_src = [
                transform_geom(b_crs, src_crs, g, precision=6) for g in geoms
            ]
        else:
            geoms_in_src = geoms

        # Robust clip: filled=True ensures outside boundary becomes nodata_value
        out_image, out_transform = mask(
            src,
            geoms_in_src,
            crop=True,
            filled=True,
            nodata=nodata_value,
            all_touched=False,
        )

        # mask() already filled; the key is to force dtype + nodata consistency
        if out_image.dtype != src.dtypes[0]:
            out_image = out_image.astype(src.dtypes[0], copy=False)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src_crs,
                "nodata": nodata_value,
            }
        )

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
    forcing_dir: Path, boundary_geometry, boundary_crs: Union[str, dict] = "EPSG:4326"
) -> Dict[Path, Path]:
    """
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
    all_tifs = sorted(forcing_dir.glob("*.tif"))
    originals = [p for p in all_tifs if not p.stem.endswith("_clipped")]
    clipped_only = [p for p in all_tifs if p.stem.endswith("_clipped")]

    # Already clipped from a prior run — originals were deleted, only _clipped files remain
    if not originals and clipped_only:
        print("Forcing rasters already clipped from a prior run — reusing existing clipped files.\n")
        mapping = {}
        for p in clipped_only:
            # Reconstruct the original key name (without _clipped suffix)
            orig_name = p.stem[: -len("_clipped")] + p.suffix
            mapping[forcing_dir / orig_name] = p
        return mapping

    if not originals:
        raise FileNotFoundError(f"No .tif forcing rasters found in: {forcing_dir}")

    geoms, b_crs = _ensure_list_of_geoms_and_crs(
        boundary_geometry, boundary_crs=boundary_crs
    )
    if not geoms:
        raise ValueError("boundary_geometry is empty; cannot clip forcings.")

    # Check bbox overlap for ALL rasters first
    non_overlapping = []

    for tif in originals:
        with rasterio.open(tif) as src:
            if src.crs is None:
                non_overlapping.append(tif)
                continue

            b_crs_local = b_crs if b_crs is not None else CRS.from_user_input(boundary_crs)

            # Transform boundary into raster CRS before overlap check
            if src.crs != b_crs_local:
                geoms_in_src = [
                    transform_geom(b_crs_local, src.crs, g, precision=6) for g in geoms
                ]
            else:
                geoms_in_src = geoms

            boundary_bbox = _union_bounds(geoms_in_src)
            raster_bbox = _normalize_bbox(
                (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            )

            if not _bbox_overlaps(boundary_bbox, raster_bbox):
                non_overlapping.append(tif)

    # If any do not overlap, print message and skip clipping entirely
    if non_overlapping:
        missing = "\n".join([f"  - {p.name}" for p in non_overlapping])
        print(
            f"clip_boundary does not overlap with the following forcing raster(s) — "
            f"skipping all clipping and using original forcings:\n{missing}\n"
        )
        return {}

    # Clip all rasters
    mapping = {}
    for tif in originals:
        new_path = clip_raster_inplace_to_boundary(
            raster_path=tif,
            boundary_geometry=boundary_geometry,
            boundary_crs=boundary_crs,
            nodata_value=0,
        )
        mapping[tif] = new_path

    return mapping


def clip_fim_to_boundary(
    fim_raster_path: Path,
    boundary_geometry,
    boundary_crs: Union[str, dict] = "EPSG:4326",
    nodata_value: int = 0,
) -> Path:
    fim_raster_path = Path(fim_raster_path)
    clipped_path = fim_raster_path.with_name(
        f"{fim_raster_path.stem}_clipped{fim_raster_path.suffix}"
    )

    geoms, b_crs = _ensure_list_of_geoms_and_crs(
        boundary_geometry, boundary_crs=boundary_crs
    )
    if not geoms:
        raise ValueError("boundary_geometry is empty; cannot clip.")

    with rasterio.open(fim_raster_path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError(f"Raster has no CRS: {fim_raster_path}")

        if b_crs is None:
            b_crs = src_crs

        if src_crs != b_crs:
            geoms_in_src = [
                transform_geom(b_crs, src_crs, g, precision=6) for g in geoms
            ]
        else:
            geoms_in_src = geoms

        boundary_bbox = _union_bounds(geoms_in_src)
        raster_bbox = _normalize_bbox(
            (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        )
        if not _bbox_overlaps(boundary_bbox, raster_bbox):
            print(
                f"clip_boundary does not overlap with FIM raster {fim_raster_path.name} — "
                f"skipping FIM clipping and using original.\n"
            )
            return fim_raster_path

        out_image, out_transform = mask(
            src,
            geoms_in_src,
            crop=True,
            filled=True,
            nodata=nodata_value,
            all_touched=False,
        )

        if out_image.dtype != src.dtypes[0]:
            out_image = out_image.astype(src.dtypes[0], copy=False)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src_crs,
                "nodata": nodata_value,
            }
        )

    with rasterio.open(clipped_path, "w", **out_meta) as dst:
        dst.write(out_image)

    try:
        compress_tif_lzw(clipped_path)
    except Exception:
        pass

    return clipped_path


# PREPROCESS THE OWP HAND BASED FIM FOR SM
def _real_inundation_dir(huc_id) -> Path:
    _, _, output_dir = setup_directories()
    return Path(output_dir) / f"flood_{huc_id}" / f"{huc_id}_inundation"


def _expected_fim_filename(huc_id, event_date) -> str:
    dt = pd.to_datetime(event_date)
    return f"NWM_{dt.strftime('%Y%m%d%H%M%S')}_{huc_id}_inundation.tif"


def prepare_FORCINGs(
    huc_id,
    event_date=None,
    data="retrospective",
    forecast_range=None,
    forecast_date=None,
    sort_by=None,
    clip_boundary=None,
    clip_boundary_crs: Union[str, dict] = "EPSG:4326",
):

    # GET FORCINGS
    print("Downloading forcings from the S3 bucket...\n")
    get_forcings(huc_id)
    print("Forcings downloaded successfully.\n")

    forcing_dir = Path(f"./HUC{huc_id}_forcings")

    mapping = {}
    did_clip_forcings = False
    if clip_boundary is not None:
        print("Boundary provided. Checking overlap with all forcings...\n")
        mapping = clip_all_forcings_if_boundary_overlaps(
            forcing_dir=forcing_dir,
            boundary_geometry=clip_boundary,
            boundary_crs=clip_boundary_crs,
        )
        if mapping:
            did_clip_forcings = True
            print("All forcing rasters clipped successfully.\n")
        else:
            print("Skipping forcing clipping due to non-overlap.\n")

    # GET THE FIM FILES
    inundation_dir = _real_inundation_dir(huc_id)

    if event_date is None:
        # No date — use whatever already exists in the output folder
        if inundation_dir.exists():
            fim_files = sorted(f for f in inundation_dir.glob("*.tif") if "processing" not in f.parts)
            if fim_files:
                print(
                    f"No event_date provided. Found {len(fim_files)} existing FIM file(s) "
                    f"in {inundation_dir} — skipping generation.\n"
                )
            else:
                raise FileNotFoundError(
                    f"No event_date provided and no FIM files found in {inundation_dir}. "
                    "Please pass event_date to generate the FIM first."
                )
        else:
            raise FileNotFoundError(
                f"No event_date provided and FIM output directory does not exist: {inundation_dir}. "
                "Please pass event_date to generate the FIM first."
            )

    elif data == "retrospective":
        dates = [event_date] if isinstance(event_date, str) else list(event_date)

        inundation_dir.mkdir(parents=True, exist_ok=True)

        dates_to_generate = []
        for d in dates:
            expected_file = inundation_dir / _expected_fim_filename(huc_id, d)
            if expected_file.exists():
                print(f"FIM already exists for {d} — reusing {expected_file.name}\n")
            else:
                dates_to_generate.append(d)

        if dates_to_generate:
            print(
                f"Generating FIM for {len(dates_to_generate)} date(s): "
                f"{', '.join(str(d) for d in dates_to_generate)}\n"
            )
            get_LFFIM(
                huc_id,
                event_date=dates_to_generate,
                data=data,
                forecast_range=forecast_range,
                forecast_date=forecast_date,
                sort_by=sort_by,
            )
            print("FIM files generated successfully.\n")
        else:
            print("All requested FIM files already exist — skipping generation.\n")

        fim_files = sorted(f for f in inundation_dir.glob("*.tif") if "processing" not in f.parts)

    else:
        # Forecast — unchanged behaviour
        print(f"Generating the FIM files for {data} event...\n")
        get_LFFIM(
            huc_id,
            event_date=event_date,
            data=data,
            forecast_range=forecast_range,
            forecast_date=forecast_date,
            sort_by=sort_by,
        )
        print("FIM files generated successfully.\n")
        fim_files = sorted(f for f in inundation_dir.glob("*.tif") if "processing" not in f.parts)

    # PREPROCESSING THE FIM FILES
    print("Preprocessing the FIM files...\n")

    # Get the HUC8 boundary
    HUC_boundary = getHUC8BoundaryByID(huc_id)

    lulc_original = forcing_dir / f"LULC_HUC{huc_id}.tif"
    reference_dir = mapping.get(lulc_original, lulc_original)

    processing_dir = inundation_dir / "processing"

    for FIM in fim_files:
        fim_name = FIM.stem
        if did_clip_forcings and clip_boundary is not None:
            FIM_finaldir = forcing_dir / f"hand_{fim_name}_clipped.tif"
        else:
            FIM_finaldir = forcing_dir / f"hand_{fim_name}.tif"

        if FIM_finaldir.exists():
            print(f"Preprocessed FIM already exists, skipping: {FIM_finaldir.name}\n")
            continue

        processing_dir.mkdir(parents=True, exist_ok=True)

        # Reproject the FIM file
        FIM_file = processing_dir / f"{FIM.stem}_reprojected.tif"
        reproject_raster(FIM, FIM_file)
        compress_tif_lzw(FIM_file)

        # Convert to binary
        out_dir_binary = processing_dir / f"{FIM.stem}_binary.tif"
        raster2binary(FIM_file, HUC_boundary, out_dir_binary)
        compress_tif_lzw(out_dir_binary)

        # Mask and clip with PWB
        final_raster = processing_dir / f"{FIM.stem}_final.tif"
        mask_with_PWB(out_dir_binary, final_raster, HUC_boundary)
        compress_tif_lzw(final_raster)

        # If boundary clipping happened for forcings, clip the FIM as well
        final_for_alignment = final_raster
        if did_clip_forcings and clip_boundary is not None:
            final_for_alignment = clip_fim_to_boundary(
                fim_raster_path=final_raster,
                boundary_geometry=clip_boundary,
                boundary_crs=clip_boundary_crs,
                nodata_value=0,
            )

        # Align final FIM raster with reference raster
        align_raster(final_for_alignment, reference_dir, FIM_finaldir)

    # Clean up processing temp files
    if processing_dir.exists() and processing_dir.is_dir():
        shutil.rmtree(processing_dir)

    print("FIM file preprocessed successfully.\n")
