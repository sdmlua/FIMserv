import fimserve as fm
import pandas as pd

#Look for the benchmark FIM data for the HUC8 and event date
def test_bm_fimlookup():
    out = fm.fim_lookup(
        HUCID="10170203",
        date_input="2019-09-19 16:00:00", #If user is more specific then they can pass date (with hour if known) along with HUC8
        # run_handfim=True,  #It will look for the owp hand fim for the mentioned HUC8 and date, if not found it will download and generate the owp hand fim
        # file_name="PSS_1_0m_20190919T165541_963659W424518N_BM.tif", #If user pass the specific filename, it will download that and assume that this is the right benchmark, else based on exact match of date it will look for the benchmark
        # start_date="2024-06-20", #If user is not sure of the exact date then they can pass a range of dates
        # end_date="2024-06-25",
    )
    print(out)  
