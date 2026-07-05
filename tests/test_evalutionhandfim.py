import fimserve as fm


# Look for the benchmark FIM data for the HUC8 and event date
def test_bm_fimlookup():
    out = fm.fim_lookup(
        # HUCID="03040204",
        # date_input="2024-06-24",  # If user is more specific then they can pass date (with hour if known) along with HUC8
        # start_date="2016-09-20",  # If user is not sure of the exact date then they can pass a range of dates
        # end_date="2016-10-09",
        # return_period=500,  # For Tier_4; BLE based benchmark, user can pass return period to genrate OWP HAND FIM to evaluate, we have 100 and 500 yr flows saves in AWS
        # tier = "tier3" , # It further filters the results- could be HWM, tier1-4
        run_handfim=True,  # It will look for the owp hand fim for the mentioned HUC8 and date, if not found it will download and generate the owp hand fim; default is False
        file_name="HWM_10_0m_20160928_20161009_791303W341522N_BM.tif",  # If user pass the specific filename, it will download that and assume that this is the right benchmark, else based on exact match of date it will look for the benchmark
        # huc_intersectedarea= True, #gives the overlapping percentage of benchmark data with each of interesected HUC
        # huc_thresholdarea= 10, #some percentage--> below which the intersected HUCs will not be considered for evaluation, default is 0
        # eval_individual_huc=False,  # If True, it will create separate folders for each HUC8 with the respective benchmark data and evaluation results; if False, it will create a combined folder with all intersecting HUC8s and the benchmark data; default is False
        out_dir="../test_FIMeval",  # Required if user wants to download the benchmark FIM data
    )
    print(out)


# # After finalizing the benchmark FIM data user can run evaluation
# def test_run_fimeval():
#     fm.run_evaluation(
#         Main_dir="../test_FIMeval",  # If user uses their own input directory where FIM outputs; basically out_dir in fim_lookup is Main_dir here
#         output_dir=None,  # Folder where evaluation results will be saved
#         shapefile_path=None,  # AOI shapefile or vector file used to clip data during evaluation. Internally uses the geopackage within folder.
#         PWB_dir=None,  # Directory containing the Permanent Water Bodies.
#         building_footprint=None,  # Local building footprint dataset (GeoJSON/Shapefile) for building-level exposure evaluation., else it will use the arcgis online hosted building footprint using REST API
#         target_crs=None,  # CRS to reproject FIM rasters to (e.g., "EPSG:3857").
#         target_resolution=None,  # Output raster resolution (units depend on CRS).
#         method_name=None,  # By default it will use 'AOI'; to explore different methods pass here
#         print_graphs=True,  # If True, generates and saves contingency maps and evaluation metric plots.
#         Evalwith_BF=True,  # If True, run evaluation with building footprint
#     )
