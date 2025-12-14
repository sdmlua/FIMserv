import fimserve as fm
import pandas as pd


# Look for the benchmark FIM data for the HUC8 and event date
def test_bm_fimlookup():
    out = fm.fim_lookup(
        HUCID="07070005",
        date_input="2019-05-30 23:00:00",  # If user is more specific then they can pass date (with hour if known) along with HUC8
        run_handfim=True,  # It will look for the OWP HAND FIM for the mentioned HUC8 and date; if not found it will download and generate the OWP HAND FIM
        file_name="S1A_9_6m_20190530T23573_910244W430506N_BM.tif",  # If user passes a specific filename, it will download that and assume it is the right benchmark
        out_dir="./FIMserv/test",  # If user wants to save the benchmark FIM in a specific directory
        # start_date="2024-06-20",  # If user is not sure of the exact date then they can pass a range of dates
        # end_date="2024-06-25",
    )
    print(out)


# After finalizing the benchmark FIM data user can run evaluation
def test_run_fimeval():
    fm.run_evaluation(
        Main_dir="./FIMserv/test",   # If user uses their own input directory where FIM outputs; basically out_dir in fim_lookup is Main_dir here
        output_dir=None,             # Folder where evaluation results will be saved
        shapefile_path=None,         # AOI shapefile or vector file used to clip data during evaluation. Internally uses the geopackage within folder.
        PWB_dir=None,                # Directory containing the Permanent Water Bodies.
        building_footprint="./AOI/building_footprint.gpkg",  # Local building footprint dataset (GeoJSON/Shapefile) for building-level exposure evaluation.
        target_crs=None,             # CRS to reproject FIM rasters to (e.g., "EPSG:3857").
        target_resolution=None,      # Output raster resolution (units depend on CRS).
        method_name=None,            # By default it will use 'AOI'; to explore different methods pass here
        countryISO=None,             # ISO-3 country code used only when downloading footprints from GEE.
        geeprojectID=None,           # Google Earth Engine project ID for footprint download (if no local file provided).
        print_graphs=True,           # If True, generates and saves contingency maps and evaluation metric plots.
        Evalwith_BF=True,            # If True, run evaluation with building footprint
    )
