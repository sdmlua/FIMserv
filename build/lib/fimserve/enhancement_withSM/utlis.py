import os
import rasterio

#INITIALIZE IN HUC EVENT DICT
def initialize_huc_event(huc_id, event_times):
    if isinstance(event_times, str):
        event_times = [event_times]
    return {huc_id: event_times}

#Create a folder and make it a working directory
def createCWD(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    os.chdir(folder_name)
    new_path = os.getcwd()
    return new_path

def compress_tif_lzw(tif_path):
    # Read original file
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        data = src.read()
    profile.update(compress='lzw')

    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(data)


