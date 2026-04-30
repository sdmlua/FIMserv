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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from .interactS3 import getHUC8BoundaryByID, get_population_GRID

FLOOD_COLOR = "#0000FF"   # blue — used for both overlay and legend patch
_FLOOD_RGB  = (0x00 / 255, 0x00 / 255, 0xFF / 255)

def _adaptive_hexbin_gridsize(extent, n_hex_across=50):
    """
    Return (nx, ny) targeting n_hex_across hexagons along the x-axis, with ny
    derived so hexagons are equilateral in metres regardless of lon/lat extent.

    Works at any scale: local (km) to continental (1000s km) — hex physical
    width simply scales with the map area.
    """
    avg_lat = (extent[2] + extent[3]) / 2
    m_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    m_per_deg_lat = 111320

    map_width_m  = (extent[1] - extent[0]) * m_per_deg_lon
    map_height_m = (extent[3] - extent[2]) * m_per_deg_lat

    nx = max(5, n_hex_across)
    hex_width_m = map_width_m / nx
    ny = max(5, int(map_height_m / hex_width_m))
    return nx, ny


def _discrete_norm_and_cmap(hex_values, n_bins=6):
    """
    Build a discrete BoundaryNorm with n_bins steps and human-readable labels.
    Uses a diverging blue→white→red palette sampled to n_bins discrete colours.
    Returns (norm, cmap, bounds, tick_labels).
    """
    valid = hex_values[hex_values > 0]
    vmin = float(valid.min()) if valid.size else 0.0
    vmax = float(hex_values.max()) if hex_values.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Round breakpoints to nice numbers matching the data magnitude
    raw = np.linspace(vmin, vmax, n_bins + 1)
    span = vmax - vmin
    magnitude = 10 ** max(0, int(np.floor(np.log10(max(span, 1)))) - 1)
    bounds = np.unique(np.round(raw / magnitude) * magnitude)
    # Ensure we always have at least 2 distinct bounds
    if len(bounds) < 2:
        bounds = np.array([vmin, vmax])

    n_colors = len(bounds) - 1
    # Sample n_colors from the upper (warm) half of RdBu_r: white→red
    # This gives green/teal at low, dark red at high — distinct and print-safe
    base = cm.get_cmap("RdYlGn_r")
    colors = [base(0.15 + 0.85 * i / max(n_colors - 1, 1)) for i in range(n_colors)]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, ncolors=n_colors)

    def _fmt(v):
        if v >= 1000:
            return f"{v/1000:.1f}k" if v % 1000 else f"{int(v//1000)}k"
        return f"{int(v)}" if v == int(v) else f"{v:.2g}"

    tick_labels = [_fmt(b) for b in bounds]
    return norm, cmap, bounds, tick_labels


def _scalebar_for_extent(boundary_gdf_4326):
    """Compute scalebar size in degrees and a human-readable label from a boundary."""
    boundary_5070 = boundary_gdf_4326.to_crs("EPSG:5070")
    bounds_5070 = boundary_5070.total_bounds
    map_width_m = bounds_5070[2] - bounds_5070[0]
    raw_length = map_width_m * 0.1
    rounded_m = int(raw_length // 500) * 500
    if rounded_m == 0:
        rounded_m = 500
    if rounded_m < 10000:
        scale_length = rounded_m
        scale_label = f"{scale_length} m"
    else:
        scale_length = int(rounded_m // 1000) * 1000
        scale_label = f"{scale_length // 1000} km"
    scalebar_size_deg = scale_length / 111000
    return scalebar_size_deg, scale_label


def _add_map_furniture(ax, fig, hb, extent, boundary, cb_label, count_label,
                       scalebar_size_deg, scale_label, cb_bounds=None, cb_tick_labels=None):
    """Attach colorbar, grid, ticks, scalebar, north arrow and count annotation to ax."""
    cb = fig.colorbar(hb, ax=ax, shrink=0.45, aspect=20, pad=0.01)
    cb.set_label(cb_label, fontsize=11)
    cb.ax.tick_params(labelsize=9)
    if cb_bounds is not None:
        cb.set_ticks(cb_bounds)
        cb.set_ticklabels(cb_tick_labels if cb_tick_labels else [str(b) for b in cb_bounds])

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, linestyle="-.", linewidth=0.3, color="gray", zorder=0)

    x_off = 0.08 * (extent[1] - extent[0])
    y_off = 0.03 * (extent[3] - extent[2])
    x_ticks = np.linspace(extent[0] + x_off, extent[1], 4)
    y_ticks = np.linspace(extent[2] + y_off, extent[3], 4)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:.2f}°W" for x in x_ticks], fontsize=11)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}°N" for y in y_ticks], fontsize=11, rotation=90, va="center")

    scalebar = AnchoredSizeBar(
        ax.transData, scalebar_size_deg, scale_label, "lower right",
        pad=0.3, color="black", frameon=True, size_vertical=0.002,
        fontproperties=fm.FontProperties(size=10),
    )
    scalebar.patch.set_facecolor("white")
    scalebar.patch.set_alpha(0.9)
    scalebar.patch.set_edgecolor("none")
    scalebar.patch.set_linewidth(0)
    ax.add_artist(scalebar)

    arrow_x = extent[1] - 0.05 * (extent[1] - extent[0])
    arrow_y = extent[3] - 0.08 * (extent[3] - extent[2])
    ax.annotate(
        "N", xy=(arrow_x, arrow_y), ha="center", va="center",
        fontsize=13, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )

    flood_patch = Patch(facecolor=FLOOD_COLOR, edgecolor=FLOOD_COLOR,
                        label="Flood Inundation Extent", linewidth=0)
    ax.legend(handles=[flood_patch], loc="lower left", fontsize=11,
              frameon=True, framealpha=0.85, edgecolor="none",
              handlelength=1.2, handleheight=1.2)
    ax.text(
        0.02, 0.98, count_label, transform=ax.transAxes, fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"),
    )

    boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, zorder=3)


def get_population_exposure(boundary_gdf, flood_map, pop_array, pop_meta, huc_id=None):
    boundary_4326 = boundary_gdf.to_crs("EPSG:4326")

    with rasterio.open(flood_map) as flood_src:
        flood_crs = flood_src.crs
        # Reproject boundary to match the flood raster CRS before masking
        geoms = [mapping(geom) for geom in boundary_4326.to_crs(flood_crs).geometry]
        flood_data_clipped, flood_transform = mask(flood_src, geoms, crop=True)
        flood_data = flood_data_clipped[0]
        flood_bounds_native = rasterio.transform.array_bounds(
            flood_data.shape[0], flood_data.shape[1], flood_transform
        )
        flood_shape = flood_data.shape

    # Convert flood bounds to 4326 for plotting (extent must be in same CRS as boundary overlay)
    from rasterio.warp import transform_bounds as _transform_bounds
    flood_bounds = _transform_bounds(flood_crs, "EPSG:4326", *flood_bounds_native) \
        if flood_crs.to_epsg() != 4326 else flood_bounds_native
    boundary = boundary_4326

    pop_transform = pop_meta["transform"]
    pop_crs = pop_meta["crs"]

    pop_array_work = pop_array.astype(np.float64)
    if pop_crs.to_epsg() != flood_crs.to_epsg():
        from rasterio.warp import calculate_default_transform, reproject as _reproject
        print(f"Auto-reprojecting population data from {pop_crs} → {flood_crs}")
        h, w = pop_array_work.shape
        t2, w2, h2 = calculate_default_transform(pop_crs, flood_crs, w, h,
                                                  *rasterio.transform.array_bounds(h, w, pop_transform))
        reproj = np.zeros((h2, w2), dtype=np.float64)
        _reproject(source=pop_array_work, destination=reproj,
                   src_transform=pop_transform, src_crs=pop_crs,
                   dst_transform=t2, dst_crs=flood_crs,
                   resampling=Resampling.nearest)
        pop_array_work = reproj
        pop_transform = t2

    # Replace NaN nodata with 0 before reprojection to prevent NaN spreading
    pop_clean = np.nan_to_num(pop_array_work, nan=0.0)

    pop_data_resampled = np.zeros(flood_shape, dtype=np.float64)
    reproject(
        source=pop_clean,
        destination=pop_data_resampled,
        src_transform=pop_transform,
        src_crs=pop_crs,
        dst_transform=flood_transform,
        dst_crs=flood_crs,
        resampling=Resampling.nearest,
    )

    # Scale pop counts to flood pixel area — convert pixel sizes to metres for any CRS
    from pyproj import Transformer, CRS as ProjCRS
    def _pixel_size_m(crs, transform, nrows):
        if ProjCRS.from_user_input(crs).is_geographic:
            avg_lat = transform.f + transform.e * nrows / 2
            return abs(transform.a) * 111320 * np.cos(np.radians(avg_lat))
        return abs(transform.a)

    pop_nrows = pop_array_work.shape[0] if pop_array_work.ndim == 2 else pop_array_work.shape[-2]
    pop_res_m   = _pixel_size_m(pop_crs,   pop_transform,  pop_nrows)
    flood_res_m = _pixel_size_m(flood_crs, flood_transform, flood_shape[0])
    pop_data_resampled *= (flood_res_m / pop_res_m) ** 2

    exposed_population = np.where(flood_data > 0, np.maximum(pop_data_resampled, 0.0), 0.0)
    total_exposed = exposed_population.sum()
    print(f"Total exposed population:\n------------------------\n{total_exposed:.1f}")

    row_inds, col_inds = np.where(exposed_population > 0)
    xs_nat, ys_nat = rasterio.transform.xy(flood_transform, row_inds, col_inds, offset="center")
    # Convert point coords to 4326 for plotting (extent is already in 4326)
    if flood_crs.to_epsg() != 4326:
        _tr = Transformer.from_crs(flood_crs, "EPSG:4326", always_xy=True)
        xs, ys = np.array(_tr.transform(xs_nat, ys_nat))
    else:
        xs, ys = np.array(xs_nat), np.array(ys_nat)
    values = exposed_population[row_inds, col_inds]

    extent = [flood_bounds[0], flood_bounds[2], flood_bounds[1], flood_bounds[3]]

    # Clip boundary to flood extent for scalebar computation
    from shapely.geometry import box
    flood_box = gpd.GeoDataFrame(geometry=[box(*flood_bounds)], crs="EPSG:4326")
    boundary_clipped = gpd.clip(boundary, flood_box)
    if boundary_clipped.empty:
        boundary_clipped = boundary
    scalebar_size_deg, scale_label = _scalebar_for_extent(boundary_clipped)

    # ~35 hexes across regardless of extent — adapts from local to continental scale
    nx, ny = _adaptive_hexbin_gridsize(extent)

    # Probe hex sums to build discrete colorbar
    fig_temp, ax_temp = plt.subplots()
    temp_hb = ax_temp.hexbin(xs, ys, C=values, reduce_C_function=np.sum,
                              gridsize=(nx, ny), extent=extent)
    hex_values = temp_hb.get_array()
    plt.close(fig_temp)

    norm, cmap, cb_bounds, cb_tick_labels = _discrete_norm_and_cmap(hex_values)

    # Flood overlay: #2166AC (blue) with alpha so it reads as clear blue on the plot
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
        cb_label="Exposed population (persons)",
        count_label=f"Exposed population: {total_exposed:.0f}",
        scalebar_size_deg=scalebar_size_deg,
        scale_label=scale_label,
        cb_bounds=cb_bounds,
        cb_tick_labels=cb_tick_labels,
    )

    fig.tight_layout()
    huc_tag = huc_id if huc_id else "unknown"
    plots_dir = Path(f"./SM_results/HUC{huc_tag}/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / f"PE_{Path(flood_map).stem}.png"
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.pause(3)
    plt.close(fig)


def getpopulation_exposure(huc_id, boundary=None, pop_raster=None):
    # Resolve boundary only when needed for clipping or plotting
    HUC_boundary = None
    if boundary is not None:
        if isinstance(boundary, (str, Path)):
            HUC_boundary = gpd.read_file(boundary).to_crs("EPSG:4326")
        elif isinstance(boundary, gpd.GeoDataFrame):
            HUC_boundary = boundary.to_crs("EPSG:4326")
        else:
            raise ValueError("boundary must be a GeoDataFrame or path to a shapefile")

    flood_dir = Path(f"./SM_results/HUC{huc_id}")
    flood_files = list(flood_dir.glob("*.tif"))

    if pop_raster is not None:
        pop_raster = Path(pop_raster)
        if not pop_raster.exists():
            raise FileNotFoundError(f"Population raster not found: {pop_raster}")
        print(f"Using user-supplied population raster: {pop_raster}")

        # Determine the flood CRS so we can reproject the pop raster to match
        flood_files_for_crs = list(flood_dir.glob("*.tif"))
        flood_crs_target = None
        if flood_files_for_crs:
            with rasterio.open(flood_files_for_crs[0]) as _fs:
                flood_crs_target = _fs.crs

        with rasterio.open(pop_raster) as src:
            pop_src_crs = src.crs

            # Reproject to flood CRS if they differ
            if flood_crs_target is not None and pop_src_crs != flood_crs_target:
                from rasterio.warp import calculate_default_transform, reproject as _reproject
                print(f"Reprojecting population raster from {pop_src_crs} to {flood_crs_target}")
                transform_reproj, width_reproj, height_reproj = calculate_default_transform(
                    pop_src_crs, flood_crs_target, src.width, src.height, *src.bounds
                )
                reproj_array = np.zeros((src.count, height_reproj, width_reproj), dtype=np.float64)
                _reproject(
                    source=src.read().astype(np.float64),
                    destination=reproj_array,
                    src_transform=src.transform,
                    src_crs=pop_src_crs,
                    dst_transform=transform_reproj,
                    dst_crs=flood_crs_target,
                    resampling=Resampling.nearest,
                )
                meta = src.meta.copy()
                meta.update({"crs": flood_crs_target, "transform": transform_reproj,
                             "width": width_reproj, "height": height_reproj, "dtype": "float64"})
                work_array = reproj_array
            else:
                work_array = src.read().astype(np.float64)
                meta = src.meta.copy()

            if HUC_boundary is not None:
                # Extra boundary provided — clip before analysis
                boundary_reproj = HUC_boundary.to_crs(meta["crs"])
                geoms = [mapping(geom) for geom in boundary_reproj.geometry]
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    with rasterio.open(tmp_path, "w", **{**meta, "count": work_array.shape[0]}) as tmp_ds:
                        tmp_ds.write(work_array)
                    with rasterio.open(tmp_path) as tmp_ds:
                        clipped, out_transform = mask(tmp_ds, geoms, crop=True)
                finally:
                    os.unlink(tmp_path)
                meta.update({"height": clipped.shape[1], "width": clipped.shape[2], "transform": out_transform})
                data_array = clipped
            else:
                data_array = work_array
    else:
        data_array, meta = get_population_GRID(HUC_boundary, huc_id=huc_id)

    # Fetch boundary from S3 only if still needed for plotting
    if HUC_boundary is None:
        HUC_boundary = gpd.GeoDataFrame(geometry=getHUC8BoundaryByID(huc_id), crs="EPSG:4326")

    for flood_map in flood_files:
        print(f"Processing population exposure for: {flood_map}")
        get_population_exposure(
            boundary_gdf=HUC_boundary,
            flood_map=flood_map,
            pop_array=data_array[0],
            huc_id=huc_id,
            pop_meta=meta,
        )
