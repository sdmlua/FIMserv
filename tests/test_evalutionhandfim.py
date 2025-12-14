import fimserve as fm
import pandas as pd


# Look for the benchmark FIM data for the HUC8 and event date
def test_bm_fimlookup():
    out = fm.fim_lookup(
<<<<<<< HEAD
        HUCID="10170203",
        date_input="2019-09-19 16:00:00",  # If user is more specific then they can pass date (with hour if known) along with HUC8
        run_handfim=True,  #It will look for the owp hand fim for the mentioned HUC8 and date, if not found it will download and generate the owp hand fim
        # file_name="PSS_1_0m_20190919T165541_963659W424518N_BM.tif", #If user pass the specific filename, it will download that and assume that this is the right benchmark, else based on exact match of date it will look for the benchmark
        # out_dir = None, # If user want to save the benchmark fim in specific directory
=======
        HUCID="07070005",
        date_input="2019-05-30 23:00:00",  # If user is more specific then they can pass date (with hour if known) along with HUC8
        run_handfim=True,  #It will look for the owp hand fim for the mentioned HUC8 and date, if not found it will download and generate the owp hand fim
        file_name="S1A_9_6m_20190530T23573_910244W430506N_BM.tif", #If user pass the specific filename, it will download that and assume that this is the right benchmark, else based on exact match of date it will look for the benchmark
        out_dir = "./FIMserv/test", # If user want to save the benchmark fim in specific directory
>>>>>>> devsd
        # start_date="2024-06-20", #If user is not sure of the exact date then they can pass a range of dates
        # end_date="2024-06-25",
    )
    print(out)
    
#After finalizing the benchmark FIM data user can run evaluation
def test_run_fimeval():
    fm.run_evaluation(
<<<<<<< HEAD
        Main_dir=None,   #If user use their own input directory to where FIM outputs; basically out_dir in fim_lookup us Main_dir here
        output_dir= None, #Folder where evaluation results will be saved
        shapefile_path= None, #AOI shapefile or vector file used to clip data during evaluation. Internally used the geopackage within folder.
        PWB_dir= None, #Directory containing the Permanent Water Bodies.
        building_footprint= None, #Local building footprint dataset (GeoJSON/Shapefile) for building-level exposure evaluation.
=======
        Main_dir="./FIMserv/test",   #If user use their own input directory to where FIM outputs; basically out_dir in fim_lookup us Main_dir here
        output_dir= None, #Folder where evaluation results will be saved
        shapefile_path= None, #AOI shapefile or vector file used to clip data during evaluation. Internally used the geopackage within folder.
        PWB_dir= None, #Directory containing the Permanent Water Bodies.
        building_footprint= "./AOI/building_footprint.gpkg", #Local building footprint dataset (GeoJSON/Shapefile) for building-level exposure evaluation.
>>>>>>> devsd
        target_crs= None, #CRS to reproject FIM rasters to (e.g., "EPSG:3857").
        target_resolution= None, #Output raster resolution (units depend on CRS).
        method_name= None, #By default it will use 'AOI' which is downloaded but incase user want to explore different method they can pass here
        countryISO= None,   #ISO-3 country code used only when downloading footprints from GEE.
        geeprojectID= None, #Google Earth Engine project ID for footprint download (if no local file provided).
<<<<<<< HEAD
        print_graphs= False, #If True, generates and saves contingency maps and evaluation metric plots.
        Evalwith_BF= False,  #If user want to run evaluation with building footprint
=======
        print_graphs= True, #If True, generates and saves contingency maps and evaluation metric plots.
        Evalwith_BF= True,  #If user want to run evaluation with building footprint
>>>>>>> devsd
        )
