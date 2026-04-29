"""
Author: Supath Dhital (sdhital@crimson.ua.edu)
Date Updated: March 02, 2026
"""

import os
import sys
import glob
import shutil
import rasterio
import subprocess
from dotenv import load_dotenv
from rasterio.io import MemoryFile

from .datadownload import setup_directories


# Incase the final outcome has wrong CRS tag
def _retag_5070_lzw_inplace(tif_path: str) -> None:
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", crs="EPSG:5070", compress="lzw", tiled=True)

        with MemoryFile() as mem:
            with mem.open(**profile) as dst:
                for b in range(1, src.count + 1):
                    dst.write(src.read(b), b)
                try:
                    cmap = src.colormap(1)
                    if cmap:
                        dst.write_colormap(1, cmap)
                except Exception:
                    pass
            data = mem.read()

    tmp = tif_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, tif_path)


# Main module for the FIM execution
def runfim(code_dir, output_dir, HUC_code, data_dir, depth=False):
    original_dir = os.getcwd()
    try:
        tools_path = os.path.join(code_dir, "tools")
        src_path = os.path.join(code_dir, "src")
        os.chdir(tools_path)
        dotenv_path = os.path.join(code_dir, ".env")
        load_dotenv(dotenv_path)
        sys.path.append(src_path)
        sys.path.append(code_dir)

        HUC_code = str(HUC_code)
        HUC_dir = os.path.join(output_dir, f"flood_{HUC_code}")
        csv_path = data_dir

        discharge_basename = os.path.basename(data_dir).split(".")[0]
        inundation_dir = os.path.join(HUC_dir, f"{HUC_code}_inundation")
        temp_dir = os.path.join(inundation_dir, "temp")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        inundation_file = os.path.join(temp_dir, f"{discharge_basename}_inundation.tif")
        Command = [
            sys.executable,
            "inundate_mosaic_wrapper.py",
            "-y",
            HUC_dir,
            "-u",
            HUC_code,
            "-f",
            csv_path,
            "-i",
            inundation_file,
        ]

        if depth:
            depth_file = os.path.join(temp_dir, f"{discharge_basename}_depth.tif")
            Command += ["-d", depth_file]
        else:
            depth_file = None

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{code_dir}"

        result = subprocess.run(
            Command,
            cwd=tools_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print(result.stdout.decode())
        if result.stderr:
            print(result.stderr.decode())

        if result.returncode == 0:
            print(f"Inundation mapping for {HUC_code} completed successfully.")

            if os.path.exists(inundation_file):
                dest_file = os.path.join(
                    inundation_dir, os.path.basename(inundation_file)
                )
                os.makedirs(inundation_dir, exist_ok=True)
                try:
                    os.replace(inundation_file, dest_file)
                except Exception:
                    if os.path.exists(dest_file):
                        os.remove(dest_file)
                    shutil.move(inundation_file, dest_file)
                _retag_5070_lzw_inplace(dest_file)

            if depth and depth_file and os.path.exists(depth_file):
                dest_depth = os.path.join(inundation_dir, os.path.basename(depth_file))
                try:
                    os.replace(depth_file, dest_depth)
                except Exception:
                    if os.path.exists(dest_depth):
                        os.remove(dest_depth)
                    shutil.move(depth_file, dest_depth)
                _retag_5070_lzw_inplace(dest_depth)

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        else:
            print(f"Failed to complete inundation mapping for {HUC_code}.")

    finally:
        os.chdir(original_dir)


def runOWPHANDFIM(huc, depth=False, version=None):
    code_dir, data_dir, output_dir = setup_directories()

    inundation_dir = os.path.join(output_dir, f"flood_{huc}", f"{huc}_inundation")
    discharge = glob.glob(os.path.join(data_dir, f"*{huc}*.csv"))
    for file in discharge:
        discharge_basename = os.path.basename(file).split(".")[0]
        fim_file = os.path.join(inundation_dir, f"{discharge_basename}_inundation.tif")
        if os.path.exists(fim_file):
            print(f"FIM already exists for {discharge_basename}, skipping.")
            continue
        runfim(code_dir, output_dir, huc, file, depth=depth)
