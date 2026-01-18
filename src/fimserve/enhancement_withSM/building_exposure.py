import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
import rasterio
import os
import tempfile
import shutil
import msfootprint as msf
from pathlib import Path
from rasterio.mask import mask
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from .interactS3 import getHUC8BoundaryByID


def _ensure_boundary_path(boundary_gdf: gpd.GeoDataFrame) -> tuple[str, tempfile.TemporaryDirectory]:
    """
    Writes boundary_gdf to a temporary GeoJSON and returns (path, tmpdir_handle).
    Keep tmpdir_handle alive while downstream code runs.
    """
    tmpdir = tempfile.TemporaryDirectory()
    boundary_path = Path(tmpdir.name) / "boundary.geojson"
    boundary_gdf.to_crs("EPSG:4326").to_file(boundary_path, driver="GeoJSON")
    return str(boundary_path), tmpdir


def get_building_exposure(boundary, flood_map, building_gpkg):
    # Accept either path or GeoDataFrame
    if isinstance(boundary, (str, Path)):
        boundary = gpd.read_file(boundary).to_crs("EPSG:4326")
    else:
        boundary = boundary.to_crs("EPSG:4326")

    geoms = [mapping(geom) for geom in boundary.geometry]

    # Load and clip buildings
    buildings = gpd.read_file(building_gpkg).to_crs("EPSG:4326")
    buildings_clipped = gpd.clip(buildings, boundary)
    buildings_clipped = buildings_clipped[
        ~buildings_clipped.geometry.is_empty & buildings_clipped.geometry.is_valid
    ]

    # Compute centroids of buildings
    centroids = buildings_clipped.centroid
    centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=buildings_clipped.crs)
    centroids_gdf = gpd.sjoin(centroids_gdf, boundary, predicate="within", how="inner")

    # Open and mask flood raster
    with rasterio.open(flood_map) as flood_src:
        flood_data_clipped, flood_transform = mask(flood_src, geoms, crop=True)
        flood_crs = flood_src.crs
        flood_bounds = rasterio.transform.array_bounds(
            flood_data_clipped.shape[1], flood_data_clipped.shape[2], flood_transform
        )
        flood_data = flood_data_clipped[0]

        # Reproject centroids to match raster CRS
        centroids_raster_crs = centroids_gdf.to_crs(flood_crs)
        coords = [(pt.x, pt.y) for pt in centroids_raster_crs.geometry]

        flood_values = np.array([val[0] for val in flood_src.sample(coords)])
        flooded_mask = flood_values > 0
        flooded_centroids = centroids_raster_crs[flooded_mask]

    flooded_count = len(flooded_centroids)
    print(f"Total flooded buildings: \n------\n {flooded_count}")

    # Ensure geometry validity
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

    extent = [flood_bounds[0], flood_bounds[2], flood_bounds[1], flood_bounds[3]]

    # Dynamically calculate hexbin gridsize
    target_hex_width_m = 800
    avg_lat = (extent[2] + extent[3]) / 2
    meters_per_degree_lon = 111320 * np.cos(np.radians(avg_lat))
    map_width_deg = extent[1] - extent[0]
    map_width_m = map_width_deg * meters_per_degree_lon
    gridsize = max(5, int(map_width_m / target_hex_width_m))
    values = np.ones_like(xs)

    # Precompute counts for binning
    fig_temp, ax_temp = plt.subplots()
    temp_hb = ax_temp.hexbin(xs, ys, C=values, reduce_C_function=np.sum, gridsize=gridsize)
    hex_counts = temp_hb.get_array()
    min_value = int(np.min(hex_counts))
    max_value = int(np.max(hex_counts))
    plt.close(fig_temp)

    num_bins = 5
    if min_value == max_value:
        max_value = min_value + 1
    bounds = np.linspace(min_value, max_value, num_bins + 1).astype(int)

    base_colors = ["#00FF00", "#CCFF00", "#FFCC00", "#FF6600", "#CC0000"]
    cmap = ListedColormap(base_colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Flood raster overlay (blue)
    flood_plot = np.zeros((*flood_data.shape, 4))
    flood_plot[..., 2] = 1.0
    flood_plot[..., 3] = (flood_data > 0).astype(float)

    plt.figure(figsize=(8, 6))
    plt.imshow(flood_plot, extent=extent, origin="upper")

    hb = plt.hexbin(
        xs, ys,
        C=values,
        reduce_C_function=np.sum,
        gridsize=gridsize,
        cmap=cmap,
        norm=norm,
        mincnt=1,
        linewidths=0.1,
        edgecolors="black",
    )

    # Boundary outline
    boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7)

    # Colorbar
    cb = plt.colorbar(
        hb,
        boundaries=bounds,
        spacing="proportional",
        ticks=bounds,
        extend="max",
        shrink=0.5,
        aspect=25,
        pad=0.01,
    )
    cb.set_label("Flooded buildings count", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.set_ticks(bounds)
    cb.set_ticklabels([str(b) for b in bounds])

    # Axis and grid
    plt.grid(True, which="both", linestyle="-.", linewidth=0.3, color="gray")

    x_offset = 0.08 * (extent[1] - extent[0])
    y_offset = 0.03 * (extent[3] - extent[2])
    x_ticks = np.linspace(extent[0] + x_offset, extent[1], 4)
    y_ticks = np.linspace(extent[2] + y_offset, extent[3], 4)

    plt.xticks(x_ticks, labels=[f"{x:.2f}°W" for x in x_ticks], fontsize=12)
    plt.yticks(y_ticks, labels=[f"{y:.2f}°N" for y in y_ticks], fontsize=12, rotation=90)

    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.tick_params(axis="both", labelsize=12)

    # Legend
    flood_patch = Patch(facecolor="blue", edgecolor="blue", alpha=1, linewidth=1.5, label="Flooded area")
    legend = plt.legend(handles=[flood_patch], loc="lower left", fontsize=12, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.6)
    legend.get_frame().set_edgecolor("none")

    # Scalebar (uses EPSG:5070)
    boundary_5070 = boundary.to_crs("EPSG:5070")
    bounds_5070 = boundary_5070.total_bounds
    map_width_m = bounds_5070[2] - bounds_5070[0]
    raw_length = map_width_m * 0.1
    rounded_m = int(raw_length // 500) * 500

    if rounded_m < 10000:
        scale_length = rounded_m
        scale_label = f"{scale_length} m"
        scalebar_size_deg = scale_length / 111000
    else:
        scale_length = int(rounded_m // 1000) * 1000
        scale_label = f"{scale_length // 1000} km"
        scalebar_size_deg = scale_length / 111000

    scalebar = AnchoredSizeBar(
        plt.gca().transData,
        scalebar_size_deg,
        scale_label,
        "lower right",
        pad=0.3,
        color="black",
        frameon=True,
        size_vertical=0.002,
        fontproperties=fm.FontProperties(size=10),
    )
    scalebar.patch.set_facecolor("white")
    scalebar.patch.set_alpha(0.9)
    scalebar.patch.set_edgecolor("none")
    scalebar.patch.set_linewidth(0)
    plt.gca().add_artist(scalebar)

    # North Arrow
    arrow_x = extent[1] - 0.05 * (extent[1] - extent[0])
    arrow_y = extent[3] - 0.0001 * (extent[3] - extent[2])
    plt.annotate(
        "N",
        xy=(arrow_x, arrow_y - 0.08 * (extent[3] - extent[2])),
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )

    plt.gca().text(
        0.04,
        0.98,
        f"Flooded buildings: {flooded_count}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    plt.tight_layout()

    flood_basename = os.path.splitext(os.path.basename(flood_map))[0]
    output_dir = os.path.dirname(flood_map)
    output_filename = os.path.join(output_dir, f"BE_{flood_basename}.png")
    plt.savefig(output_filename, dpi=600)
    plt.show()


def getbuilding_exposure(huc_id, boundary=None, geeprojectID=None):
    """
    Wrapper:
      - Ensures msfootprint always receives a boundary *path*.
      - Keeps boundary as a GeoDataFrame for plotting/clip operations.
    """
    countryISO = "USA"
    out_dir = Path(f"./Results/HUC{huc_id}/BuildingFootprint")
    building_gpkg = out_dir / "building_footprint.gpkg"

    tmpdir = None
    boundary_path = None

    # Normalize boundary to GeoDataFrame & boundary_path
    if boundary is not None:
        if isinstance(boundary, (str, Path)):
            boundary_path = str(boundary)
            HUC_boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        elif isinstance(boundary, gpd.GeoDataFrame):
            HUC_boundary = boundary.to_crs("EPSG:4326")
            boundary_path, tmpdir = _ensure_boundary_path(HUC_boundary)
        else:
            raise ValueError("boundary must be a GeoDataFrame or path to a shapefile/geojson")
    else:
        HUC_geojson = getHUC8BoundaryByID(huc_id)  # likely GeoSeries
        HUC_boundary = gpd.GeoDataFrame(geometry=HUC_geojson, crs="EPSG:4326")
        boundary_path, tmpdir = _ensure_boundary_path(HUC_boundary)

    # Run msfootprint using the boundary path (not GeoDataFrame)
    try:
        if not building_gpkg.exists():
            msf.BuildingFootprintwithISO(countryISO, boundary_path, out_dir, geeprojectID)

        # Load flood maps and compute building exposure plots
        flood_dir = Path(f"./Results/HUC{huc_id}")
        flood_files = list(flood_dir.glob("*.tif"))

        for flood_map in flood_files:
            print(f"Processing building exposure for: {flood_map}")
            get_building_exposure(HUC_boundary, str(flood_map), str(building_gpkg))

    finally:
        # Cleanup temp boundary file
        if tmpdir is not None:
            tmpdir.cleanup()

    # Cleanup msfootprint outputs
    if out_dir.exists():
        shutil.rmtree(out_dir)
