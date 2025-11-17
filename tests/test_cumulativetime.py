import fimserve as fm
import pandas as pd
import time

# Load HUC8 area data into a DataFrame
area_data = {
    "HUC8": [
        "11140102",
        "12020007",
        "11130304",
        "11130202",
        "11090203",
        "11140104",
        "12030109",
        "11130203",
        "12020006",
        "12030105",
        "11050003",
        "12100201",
        "11130301",
        "11070103",
        "08040101",
        "12020004",
        "11010014",
        "11090202",
        "12020001",
        "11090204",
        "12090301",
        "11130303",
        "12060102",
        "11130302",
    ],
    "Area_sqkm": [
        1769.42,
        1818.87,
        1893.67,
        2071.14,
        2530.83,
        2597.43,
        2784.19,
        2848.79,
        2899.44,
        3547.53,
        3586.18,
        3710.7,
        3813.95,
        3930.32,
        4010.01,
        4185.94,
        4666.06,
        4747.2,
        5036.16,
        5130.27,
        5688.12,
        6495.86,
        7243.48,
        8310.93,
    ],
}
area_df = pd.DataFrame(area_data)

# For multiple HUC8s
huc_event_dict = {huc: ["2016-10-09 15:00:00"] for huc in area_df["HUC8"]}

area_df = area_df.sort_values("Area_sqkm")

# For recording cumulative metrics
results = []
cumulative_area = 0.0
cumulative_time_sec = 0.0

for index, row in area_df.iterrows():
    huc8 = row["HUC8"]
    area = row["Area_sqkm"]
    event_time = huc_event_dict[huc8][0]

    start = time.time()
    print(f"Processing HUC8: {huc8}")

    fm.DownloadHUC8(huc8)
    fm.getNWMretrospectivedata(huc_event_dict={huc8: [event_time]})
    fm.runOWPHANDFIM(huc8)

    end = time.time()
    duration_sec = end - start

    cumulative_area += area
    cumulative_time_sec += duration_sec

    results.append(
        {
            "HUC8": huc8,
            "Area_sqkm": area,
            "EventTime": event_time,
            "TimeTaken_sec": round(duration_sec, 2),
            "CumulativeArea_sqkm": round(cumulative_area, 2),
            "CumulativeTime_sec": round(cumulative_time_sec, 2),
        }
    )

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("./cumulative_time.csv", index=False)
print("Saved cumulative results to 'cumulative_fim_results_sec.csv'")
