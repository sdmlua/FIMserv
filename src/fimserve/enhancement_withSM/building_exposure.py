import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping, box
import rasterio
import os
import shutil
import fimeval as fe
from pathlib import Path
from rasterio.mask import mask

from .interactS3 import getHUC8BoundaryByID
from .pop_exposure import _adaptive_hexbin_gridsize, _scalebar_for_extent, _discrete_norm_and_cmap, _add_map_furniture, FLOOD_COLOR, _FLOOD_RGB


def get_building_exposure(boundary, flood_map, building_gpkg, huc_id=None):
    if isinstance(boundary, (str, Path)):
        boundary = gpd.read_file(boundary).to_crs("EPSG:4326")
    else:
        boundary = boundary.to_crs("EPSG:4326")

    geoms = [mapping(geom) for geom in boundary.geometry]

    buildings = gpd.read_file(building_gpkg).to_crs("EPSG:4326")
    buildings_clipped = gpd.clip(buildings, boundary)
    buildings_clipped = buildings_clipped[
        ~buildings_clipped.geometry.is_empty & buildings_clipped.geometry.is_valid
    ]

    centroids = buildings_clipped.centroid
    centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=buildings_clipped.crs)
    centroids_gdf = gpd.sjoin(centroids_gdf, boundary, predicate="within", how="inner")

    with rasterio.open(flood_map) as flood_src:
        flood_data_clipped, flood_transform = mask(flood_src, geoms, crop=True)
        flood_crs = flood_src.crs
        flood_bounds = rasterio.transform.array_bounds(
            flood_data_clipped.shape[1], flood_data_clipped.shape[2], flood_transform
        )
        flood_data = flood_data_clipped[0]

        centroids_raster_crs = centroids_gdf.to_crs(flood_crs)
        coords = [(pt.x, pt.y) for pt in centroids_raster_crs.geometry]
        flood_values = np.array([val[0] for val in flood_src.sample(coords)])
        flooded_mask = flood_values > 0
        flooded_centroids = centroids_raster_crs[flooded_mask]

    flooded_count = len(flooded_centroids)
    print(f"Total flooded buildings: \n------\n {flooded_count}")

    flooded_centroids = flooded_centroids[
        flooded_centroids.geometry.notnull()
        & flooded_centroids.geometry.is_valid
        & ~flooded_centroids.geometry.is_empty
    ]

    if flooded_centroids.empty:
        print("No flooded buildings found for this flood map. Skipping plot.")
        return

    xs = flooded_centroids.geometry.x.values.astype(float)
    ys = flooded_centroids.geometry.y.values.astype(float)
    values = np.ones_like(xs)

    extent = [flood_bounds[0], flood_bounds[2], flood_bounds[1], flood_bounds[3]]

    # Clip boundary to flood extent for scalebar
    flood_box = gpd.GeoDataFrame(geometry=[box(*flood_bounds)], crs="EPSG:4326")
    boundary_clipped = gpd.clip(boundary, flood_box)
    if boundary_clipped.empty:
        boundary_clipped = boundary
    scalebar_size_deg, scale_label = _scalebar_for_extent(boundary_clipped)

    nx, ny = _adaptive_hexbin_gridsize(extent)

    # Probe hex counts for discrete colour normalisation
    fig_temp, ax_temp = plt.subplots()
    temp_hb = ax_temp.hexbin(xs, ys, C=values, reduce_C_function=np.sum,
                              gridsize=(nx, ny), extent=extent)
    hex_counts = temp_hb.get_array()
    plt.close(fig_temp)

    norm, cmap, cb_bounds, cb_tick_labels = _discrete_norm_and_cmap(hex_counts)

    flood_mask_2d = (flood_data > 0).astype(float)
    flood_plot = np.zeros((*flood_data.shape, 4))
    flood_plot[..., 0] = _FLOOD_RGB[0]
    flood_plot[..., 1] = _FLOOD_RGB[1]
    flood_plot[..., 2] = _FLOOD_RGB[2]
    flood_plot[..., 3] = flood_mask_2d * 0.65

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(flood_plot, extent=extent, origin="upper", zorder=1)
    hb = ax.hexbin(
        xs, ys, C=values, reduce_C_function=np.sum,
        gridsize=(nx, ny), cmap=cmap, norm=norm,
        mincnt=1, alpha=0.88, edgecolors="face", linewidths=0.0,
        zorder=2, extent=extent,
    )

    _add_map_furniture(
        ax, fig, hb, extent, boundary,
        cb_label="Flooded buildings count",
        count_label=f"Flooded buildings: {flooded_count}",
        scalebar_size_deg=scalebar_size_deg,
        scale_label=scale_label,
        cb_bounds=cb_bounds,
        cb_tick_labels=cb_tick_labels,
    )

    fig.tight_layout()
    flood_basename = os.path.splitext(os.path.basename(flood_map))[0]
    huc_tag = huc_id if huc_id else "unknown"
    plots_dir = Path(f"./SM_results/HUC{huc_tag}/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_filename = plots_dir / f"BE_{flood_basename}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.pause(3)
    plt.close(fig)


def getbuilding_exposure(huc_id, boundary=None):
    out_dir = Path(f"./SM_results/HUC{huc_id}/BuildingFootprint")
    building_gpkg = out_dir / "building_footprints.gpkg"

    if boundary is not None:
        if isinstance(boundary, (str, Path)):
            HUC_boundary = gpd.read_file(boundary).to_crs("EPSG:4326")
        elif isinstance(boundary, gpd.GeoDataFrame):
            HUC_boundary = boundary.to_crs("EPSG:4326")
    else:
        HUC_geojson = getHUC8BoundaryByID(huc_id)
        HUC_boundary = gpd.GeoDataFrame(geometry=HUC_geojson, crs="EPSG:4326")

    try:
        if not building_gpkg.exists():
            print(f"Downloading footprints via ArcGIS REST API...")
            fe.getBuildingFootprint(boundary=HUC_boundary, output_dir=out_dir)

        flood_dir = Path(f"./SM_results/HUC{huc_id}")
        flood_files = list(flood_dir.glob("*.tif"))

        for flood_map in flood_files:
            get_building_exposure(HUC_boundary, str(flood_map), str(building_gpkg), huc_id=huc_id)

    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)
