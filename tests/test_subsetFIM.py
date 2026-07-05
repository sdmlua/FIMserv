import fimserve as fm

huc = "03020202"

# If user want to subset FIM from X, Y coord
location = [-77.505826, 35.323955]
fm.subsetFIM(location, huc, method="xy")

# # If user want to subset FIM from boundary
# location = "/Users/supath/Downloads/MSResearch/CNN/fimPackage/docs/subsetBoundary/boundary.gpkg"
# fm.subsetFIM(location, huc, method="boundary")
