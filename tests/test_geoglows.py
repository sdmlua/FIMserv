import fimserve as fm
import pandas as pd

huc = "12060102"

hydrotable_dir = "/Users/supath/Downloads/MSResearch/FIMserv/FIMserv/GeoGLOWS/hydrotable/fim45geoglows_12060102.csv"

# Download the data
fm.DownloadHUC8(huc)

# Downloading the GeoGlows Streamflow data
fm.getGEOGLOWSstreamflow(
    huc,
    event_time="2016-10-15",
    hydrotable=hydrotable_dir,
)

# run the FIM model
fm.runOWPHANDFIM(huc)
