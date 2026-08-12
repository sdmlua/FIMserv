"""
Author: Supath Dhital (sdhital@crimson.ua.edu)
Date Updated: August 12, 2026

Sample usage of the NWM Analysis and Assimilation (AnA) discharge module.
Covers the aggregated, continuous hourly, event day and single timestep options.
"""

import fimserve as fm

huc = "03020202"

start_date = "2024-09-26"
end_date = "2024-09-28"

# AnA is indexed by valid time, so the range below is the period being mapped, not a
# forecast cycle. NWM AnA is only available from 2018-09-17 onwards.

# One aggregated discharge file for the whole range (default)
fm.getNWManalysisAssim(huc, start_date, end_date)

# Continuous hourly discharge, one CSV per timestep
# fm.getNWManalysisAssim(huc, start_date, end_date, continuous_discharge=True)

# Only the event day within the range, aggregated with the chosen statistic
# fm.getNWManalysisAssim(huc, start_date, end_date, value_times="2024-09-27")

# Only a single timestep within the range
# fm.getNWManalysisAssim(
#     huc, start_date, end_date, value_times="2024-09-27 12:00:00"
# )

# Several value_times at once, same as getNWMretrospectivedata accepts
# fm.getNWManalysisAssim(
#     huc,
#     start_date,
#     end_date,
#     value_times=["2024-09-27 06:00:00", "2024-09-27 18:00:00"],
# )

# Generate FIM from whichever AnA discharge files were saved
# fm.runOWPHANDFIM(huc)
