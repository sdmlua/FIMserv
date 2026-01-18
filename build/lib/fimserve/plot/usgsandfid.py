import os
import fiona
import geopandas as gpd
from tabulate import tabulate

from ..datadownload import setup_directories


def usgsandfid(gpkg_file, layer_name=None):
    if layer_name is None:
        layer_name = fiona.listlayers(gpkg_file)[0]
    gdf = gpd.read_file(gpkg_file, layer=layer_name)
    if "location_id" not in gdf.columns or "feature_id" not in gdf.columns:
        raise ValueError(
            "The GeoPackage does not contain 'location_id' or 'feature_id' columns."
        )

    gdf = gdf.rename(columns={"location_id": "USGS gauge station ID"})
    return gdf[["USGS gauge station ID", "feature_id"]]


def display_table(df):
    table = tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False)
    print(table)


def GetUSGSIDandCorrFID(huc):
    code_dir, data_dir, output_dir = setup_directories()
    gpkg_file_path = os.path.join(
        output_dir, f"flood_{huc}", f"{huc}", "usgs_subset_gages.gpkg"
    )
    table = usgsandfid(gpkg_file_path)
    print(f"***USGS gauge station ID and corresponding feature ID for HUC {huc}***")
    display_table(table)
    return table
