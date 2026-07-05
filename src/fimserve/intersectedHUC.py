import s3fs
import tempfile
from io import BytesIO
import json
import requests
import rasterio
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Union, List
from shapely.geometry import box
from shapely.ops import unary_union

# ---THIS S3 approach takes time- so it is retrieved now and used the arcgis REST API approach--
# Initialize anonymous S3 filesystem
fs = s3fs.S3FileSystem(anon=True)


# FINDING THE INTERSECTED HUC8
def find_intersecting_huc8ID(huc8_gdf, user_boundary_path):
    user_gdf = gpd.read_file(user_boundary_path)
    if user_gdf.crs != "EPSG:4326":
        user_gdf = user_gdf.to_crs("EPSG:4326")
    intersecting = gpd.overlay(huc8_gdf, user_gdf, how="intersection")

    if intersecting.empty:
        return "No HUC8 region intersects with the user-defined boundary."

    output = ["Your boundary falls within:\n----------------------------------"]
    for _, row in intersecting.iterrows():
        output.append(f"HUC8 - {row['HUC8']}\nNAME - {row['name']}\n")
    return "\n".join(output)


# GETTING THE HUC8 BOUNDARIES FROM S3
def HUC8_inS3(fs, bucket, prefix="HUC8_boundaries/"):
    # List available files
    files = fs.ls(f"{bucket}/{prefix}")
    gpkg_key = next((f for f in files if f.endswith(".gpkg")), None)

    if gpkg_key is None:
        raise FileNotFoundError(f"No .gpkg file found in s3://{bucket}/{prefix}")

    # Read file into temporary file
    with fs.open(gpkg_key, "rb") as s3file:
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            tmp_file.write(s3file.read())
            tmp_path = tmp_file.name

    gdf = gpd.read_file(tmp_path)
    return gdf


# WRAPPING ALL FUNCTIONS
def getIntersectedHUC8ID_old(user_boundary):
    bucket_name = "sdmlab"
    HUC8_gdf = HUC8_inS3(fs, bucket_name)
    HUC8 = find_intersecting_huc8ID(HUC8_gdf, user_boundary)
    return HUC8


# NEW APPROACH USING ARCGIS REST API- MUCH FASTER THAN S3 DOWNLOAD
class HUC8RESTFinder:
    def __init__(self, debug: bool = False):
        # Public ArcGIS REST Service (WBD HUC8)
        self.url = "https://services.arcgis.com/ts4gk3YgS68yLGFl/arcgis/rest/services/HUC8_Boundaries/FeatureServer/0"
        self.debug = debug
        self.target_albers = 5070  # Standard for USA Area Calculations

    def log(self, msg):
        if self.debug:
            print(f"[HUC8-LOG] {msg}")

    def _extract_geometry(
        self, path: Union[str, Path], layer: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        path = Path(path)
        ext = path.suffix.lower()

        if ext in [".tif", ".tiff", ".img", ".vrt"]:
            with rasterio.open(path) as src:
                b = src.bounds
                geom = box(b.left, b.bottom, b.right, b.top)
                return gpd.GeoDataFrame({"geometry": [geom]}, crs=src.crs)

        if ext == ".kml":
            import fiona

            fiona.drvsupport.supported_drivers["KML"] = "rw"
            return gpd.read_file(path)

        return gpd.read_file(path, layer=layer)

    def _get_rings(self, geometry) -> List:
        """Standardizes rings for ESRI JSON from Polygon or MultiPolygon."""
        if geometry.geom_type == "Polygon":
            return [list(geometry.exterior.coords)]
        elif hasattr(geometry, "geoms"):
            return [list(p.exterior.coords) for p in geometry.geoms]
        return []

    # This function calculates the percentage area of the user's boundary that overlaps with each intersecting HUC8 region.
    def get_huc_area_mapping(
        self, boundary_path: Union[str, Path], layer: Optional[str] = None
    ) -> Dict[str, float]:
        """Returns a raw dictionary {HUC8: percentage} for evaluation logic."""
        user_gdf = self._extract_geometry(boundary_path, layer)
        if user_gdf.crs is None:
            return {}

        user_4326 = user_gdf.to_crs(4326)
        user_geom_union = unary_union(user_4326.geometry)
        rings = self._get_rings(user_geom_union)

        esri_geom = {"rings": rings, "spatialReference": {"wkid": 4326}}

        params = {
            "f": "geojson",
            "where": "1=1",
            "geometry": json.dumps(esri_geom),
            "geometryType": "esriGeometryPolygon",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": 4326,
            "outFields": "HUC8",
            "returnGeometry": "true",
            "outSR": 4326,
        }

        response = requests.post(f"{self.url}/query", data=params, timeout=30)
        if response.status_code != 200:
            return {}

        huc_results = gpd.read_file(BytesIO(response.content))
        if huc_results.empty:
            return {}

        user_albers = user_gdf.to_crs(self.target_albers)
        huc_albers = huc_results.to_crs(self.target_albers)
        total_user_area = unary_union(user_albers.geometry).area

        mapping = {}
        for _, huc in huc_albers.iterrows():
            inter_area = user_albers.geometry.intersection(huc.geometry).area.sum()
            mapping[str(huc["HUC8"])] = (inter_area / total_user_area) * 100

        return mapping

    # This function provides a human-readable output of the intersecting HUC8 regions, optionally including the percentage area overlap.
    def find_intersecting_hucs(
        self,
        boundary_path: Union[str, Path],
        layer: Optional[str] = None,
        calc_area: bool = False,
    ) -> str:
        """Standard human-readable output."""
        if calc_area:
            mapping = self.get_huc_area_mapping(boundary_path, layer)
            if not mapping:
                return "No HUC8 regions found."
            output = ["Your boundary distribution across HUC8s:\n" + "-" * 35]
            for huc_id, pct in mapping.items():
                output.append(f"HUC8: {huc_id} | OVERLAP: {pct:.2f}%")
            return "\n".join(output)

        # Simple intersection logic
        user_gdf = self._extract_geometry(boundary_path, layer)
        user_4326 = user_gdf.to_crs(4326)
        rings = self._get_rings(unary_union(user_4326.geometry))

        params = {
            "f": "json",
            "where": "1=1",
            "geometry": json.dumps({"rings": rings}),
            "geometryType": "esriGeometryPolygon",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": 4326,
            "outFields": "HUC8,name",
            "returnGeometry": "false",
        }
        res = requests.post(f"{self.url}/query", data=params).json()
        features = res.get("features", [])
        if not features:
            return "No HUC8 regions found."

        output = ["Your boundary falls within:\n" + "-" * 30]
        for f in features:
            attr = f["attributes"]
            output.append(f"HUC8: {attr['HUC8']} | Name: {attr['name']}")
        return "\n".join(output)


# Main funtion
def getIntersectedHUC8ID(user_boundary, area=False, layer=None):
    finder = HUC8RESTFinder(debug=False)
    return finder.find_intersecting_hucs(user_boundary, layer=layer, calc_area=area)
