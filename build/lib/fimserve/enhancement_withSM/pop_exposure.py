import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import numpy as np
from rasterio.warp import reproject
from pathlib import Path
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

from .interactS3 import getHUC8BoundaryByID, get_population_GRID

def calculate_GRIDnSCALEbar(extent, boundary_gdf_4326): 
    avg_lat = (extent[2] + extent[3]) / 2
    meters_per_degree_lon = 111320 * np.cos(np.radians(avg_lat))
    map_width_deg = extent[1] - extent[0]
    map_width_m = map_width_deg * meters_per_degree_lon
    target_hex_width_m = 1000
    gridsize = max(5, int(map_width_m / target_hex_width_m))

    boundary_5070 = boundary_gdf_4326.to_crs("EPSG:5070") # Use the 4326 version passed in
    bounds_5070 = boundary_5070.total_bounds
    map_width_m_5070 = bounds_5070[2] - bounds_5070[0]
    raw_length = map_width_m_5070 * 0.1
    rounded_m = int(raw_length // 500) * 500

    if rounded_m < 10000:
        scale_length = rounded_m
        scale_label = f"{scale_length} m"
    else:
        scale_length = int(rounded_m // 1000) * 1000
        scale_label = f"{scale_length // 1000} km"

    scalebar_size_deg = scale_length / 111000
    return gridsize, scalebar_size_deg, scale_label


def get_population_exposure(boundary_gdf, flood_map, pop_array, pop_meta):
    boundary = boundary_gdf.to_crs("EPSG:4326")
    geoms = [mapping(geom) for geom in boundary.geometry]

    # Calculate gridsize and scalebar based on the HUC boundary's extent
    huc_bounds_4326 = boundary.total_bounds
    huc_extent_4326 = [huc_bounds_4326[0], huc_bounds_4326[2], huc_bounds_4326[1], huc_bounds_4326[3]]
    gridsize, scalebar_size_deg, scale_label = calculate_GRIDnSCALEbar(huc_extent_4326, boundary)


    with rasterio.open(flood_map) as flood_src:
        flood_data_clipped, flood_transform = mask(flood_src, geoms, crop=True)
        flood_data = flood_data_clipped[0]
        flood_crs = flood_src.crs
        flood_bounds = rasterio.transform.array_bounds(
            flood_data.shape[0], flood_data.shape[1], flood_transform
        )
        flood_shape = flood_data.shape

    pop_transform = pop_meta["transform"]
    pop_crs = pop_meta["crs"]
    assert pop_crs == flood_crs, "CRS mismatch between rasters"

    pop_data_resampled = np.empty(flood_shape, dtype=np.float32)
    reproject(
        source=pop_array,
        destination=pop_data_resampled,
        src_transform=pop_transform,
        src_crs=pop_crs,
        dst_transform=flood_transform,
        dst_crs=flood_crs,
        resampling=Resampling.bilinear
    )

    flood_mask = flood_data > 0
    pop_mask = pop_data_resampled > 0
    exposure_mask = flood_mask & pop_mask
    exposed_population = np.where(exposure_mask, pop_data_resampled, 0).astype(np.int32)
    total_exposed = exposed_population.sum()
    print(f"Total exposed population:\n------------------------\n{int(total_exposed)}")

    row_inds, col_inds = np.where(exposed_population > 0)
    xs, ys = rasterio.transform.xy(flood_transform, row_inds, col_inds, offset='center')
    values = exposed_population[row_inds, col_inds] 

    extent = [flood_bounds[0], flood_bounds[2], flood_bounds[1], flood_bounds[3]]

    fig_temp, ax_temp = plt.subplots()
    temp_hb = ax_temp.hexbin(xs, ys, C=values, reduce_C_function=np.sum, gridsize=gridsize)
    hex_values = temp_hb.get_array()
    plt.close(fig_temp)

    min_val, max_val = int(hex_values.min()), int(hex_values.max()) or 1
    bounds = np.linspace(min_val, max_val, 6).astype(int)
    cmap = ListedColormap(["#00FF00", "#CCFF00", "#FFCC00", "#FF6600", "#CC0000"])
    norm = BoundaryNorm(bounds, cmap.N)

    flood_plot = np.zeros((*flood_data.shape, 4))
    flood_plot[..., 2] = 1.0
    flood_plot[..., 3] = (flood_data > 0).astype(float)

    plt.figure(figsize=(8, 6))
    plt.imshow(flood_plot, extent=extent, origin='upper')
    hb = plt.hexbin(xs, ys, C=values, reduce_C_function=np.sum, gridsize=gridsize,
                    cmap=cmap, norm=norm, mincnt=1, edgecolors='black', linewidths=0.1)

    boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=0.7)

    cb = plt.colorbar(hb, boundaries=bounds, spacing='proportional', ticks=bounds,
                      shrink=0.6, aspect=25, pad=0.01, extend='max')
    cb.set_label('Exposed population count', fontsize=12)
    cb.ax.tick_params(labelsize=10)

    plt.grid(True, linestyle='-.', linewidth=0.3, color='gray')

    x_offset = 0.08 * (extent[1] - extent[0]) 
    y_offset = 0.03 * (extent[3] - extent[2])

    x_ticks = np.linspace(extent[0] + x_offset, extent[1], 4)
    y_ticks = np.linspace(extent[2] + y_offset, extent[3], 4)

    plt.xticks(
        x_ticks,
        labels=[f'{x:.2f}°W' for x in x_ticks],
        fontsize=12
    )
    plt.yticks(
        y_ticks,
        labels=[f'{y:.2f}°N' for y in y_ticks],
        fontsize=12,
        rotation=90
    )

    scalebar = AnchoredSizeBar(plt.gca().transData, scalebar_size_deg, scale_label, 'lower right',
                            pad=0.3, color='black', frameon=True, size_vertical=0.002,
                            fontproperties=fm.FontProperties(size=10))
    scalebar.patch.set_facecolor('white')
    scalebar.patch.set_alpha(0.9)
    scalebar.patch.set_edgecolor('none') 
    scalebar.patch.set_linewidth(0)
    plt.gca().add_artist(scalebar)
    arrow_x = extent[1] - 0.05 * (extent[1] - extent[0])
    arrow_y = extent[3] - 0.0001 * (extent[3] - extent[2])
    plt.annotate('N',
                 xy=(arrow_x, arrow_y - 0.08 * (extent[3] - extent[2])),
                 ha='center', va='center', fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    flood_patch = Patch(facecolor='blue', edgecolor='blue', alpha=1, linewidth=1.5, label='Flooded area')
    plt.legend(handles=[flood_patch], loc='lower left', fontsize=12, frameon=False)
    plt.gca().text(0.04, 0.98, f"Exposed population: {int(total_exposed)}",
                   transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout()
    output_name = f"PE_{Path(flood_map).stem}.png"
    output_path = Path(flood_map).parent / output_name
    plt.savefig(output_path, dpi=600)
    plt.show()

def getpopulation_exposure(huc_id, boundary=None):
    if boundary is not None:
        if isinstance(boundary, (str, Path)):
            HUC_boundary = gpd.read_file(boundary).to_crs("EPSG:4326")
        elif isinstance(boundary, gpd.GeoDataFrame):
            HUC_boundary = boundary.to_crs("EPSG:4326")
        else:
            raise ValueError("boundary must be a GeoDataFrame or path to a shapefile")
    else:
        HUC_geojson = getHUC8BoundaryByID(huc_id)
        HUC_boundary = gpd.GeoDataFrame(geometry=HUC_geojson, crs="EPSG:4326")

    # Load flood maps and compute building exposure
    flood_dir = Path(f"./Results/HUC{huc_id}")
    flood_files = list(flood_dir.glob("*.tif"))
    data_array, meta = get_population_GRID(HUC_boundary)
    
    for flood_map in flood_files:
        print(f"Processing population exposure for: {flood_map}")
        get_population_exposure(
            boundary_gdf=HUC_boundary,
            flood_map=flood_map,
            pop_array=data_array[0],  
            pop_meta=meta
        )

