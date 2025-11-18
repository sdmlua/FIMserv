"""
FIM evaluation workflow module within FIMserv.
Author: Supath Dhital
Date: 18 Nov, 2025
"""

import os
from pathlib import Path
from typing import Optional
import fimeval as fe # type: ignore

class run_evaluation:
    """
    Runs the full Flood Inundation Mapping (FIM) evaluation workflow.

    This class loads benchmark FIM data (from FIMbench or user-provided folders),
    performs raster-based evaluation, optionally prints summary plots, and can also
    run building-footprint-based exposure evaluation.

    The below mostly used internally, incase user want to run evaluation with modifications of data inputs they can pass different parameters
    Parameters
    ----------
    Main_dir : Optional[str]; Directory containing benchmark FIM inputs. Defaults to ./FIM_Evaluation/FIM_inputs if not provided.
    output_dir : Optional[str]; Folder where evaluation results and graphs will be saved.
    shapefile_path : Optional[str]; AOI shapefile or vector file used to clip data during evaluation. Internally used the geopackage within folder.
    PWB_dir : Optional[str]; Directory containing the Permanent Water Bodies.
    building_footprint : Optional[str]; Local building footprint dataset (GeoJSON/Shapefile/Parquet) for building-level exposure evaluation.
    target_crs : Optional[str]; CRS to reproject FIM rasters to (e.g., "EPSG:3857").
    target_resolution : Optional[float]; Output raster resolution (units depend on CRS).
    method_name : Optional[str]; Name of the evaluation method to evaluate. Defaults to "AOI".
    countryISO : Optional[str]; ISO-3 country code used only when downloading footprints from GEE.
    geeprojectID : Optional[str]; Google Earth Engine project ID for footprint download (if no local file provided).
    print_graphs : bool; If True, generates and saves contingency maps and evaluation metric plots.
    Evalwith_BF : bool; If True, performs building-footprint-based exposure evaluation.
    """
    def __init__(
        self,
        Main_dir: Optional[str] = None,   #If user use their own input directory to save benchmark FIM outputs
        output_dir: Optional[str] = None,
        shapefile_path: Optional[str] = None,
        PWB_dir: Optional[str] = None,
        building_footprint: Optional[str] = None,
        target_crs: Optional[str] = None,
        target_resolution: Optional[float] = None,
        method_name: Optional[str] = None, #By default it will use 'AOI' which is downloaded but incase user want to explore different method they can pass here
        countryISO: Optional[str] = None,
        geeprojectID: Optional[str] = None,
        print_graphs: bool = False,
        Evalwith_BF: bool = False,  #If user want to run evaluation with building footprint
        ):
        if Main_dir is None:
            self.Main_dir = os.path.join(os.getcwd(), "FIM_Evaluation", "FIM_inputs")
        else:
            self.Main_dir = Main_dir
        
        self.shapefile_path = shapefile_path
        
        if output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "FIM_Evaluation", "Evaluation_Results")
        else:
            self.output_dir = output_dir
            
        self.PWB_dir = PWB_dir
        self.building_footprint = building_footprint
        self.target_crs = target_crs
        self.target_resolution = target_resolution
        if method_name is None:
            self.method_name = 'AOI'
        else:
            self.method_name = method_name
        self.countryISO = countryISO
        self.geeprojectID = geeprojectID
        self.Evalwith_BF = Evalwith_BF
        self.print_graphs = print_graphs
        
        #run the process
        self.run_eval()
        
    def run_eval(self):
        fe.EvaluateFIM(
            main_dir=self.Main_dir,
            method_name=self.method_name,
            output_dir=self.output_dir,
            PWB_dir=self.PWB_dir,
            shapefile_dir=self.shapefile_path,
            target_crs=self.target_crs,
            target_resolution=self.target_resolution,
        )
        
        if self.print_graphs:
            fe.PrintContingencyMap(
                main_dir=self.Main_dir,
                method_name=self.method_name,
                output_dir=self.output_dir,
            )
            fe.PlotEvaluationMetrics(
                main_dir=self.Main_dir,
                method_name=self.method_name,
                output_dir=self.output_dir,
            )
        
        if self.Evalwith_BF:
            try:
                fe.EvaluationWithBuildingFootprint(
                    main_dir=self.Main_dir,
                    method_name=self.method_name,
                    output_dir=self.output_dir,
                    countryISO=self.countryISO,
                    geeprojectID=self.geeprojectID,
                    building_footprint=self.building_footprint,
                )
            except Exception as e:
                print("Skipping evaluation with building footprint due to error:", e)
            

