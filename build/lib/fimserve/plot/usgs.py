import os
import teehr
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from ..datadownload import setup_directories


def getUSGSdata(data_dir, usgs_site, start_date, end_date):
    location_id = f"usgs-{usgs_site}"

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    start_dateSTR = start_date.strftime("%Y-%m-%d")
    end_dateSTR = end_date.strftime("%Y-%m-%d")
    target_file = f"{start_dateSTR}_{end_dateSTR}.parquet"

    target_dir = os.path.join(data_dir, target_file)

    if not os.path.exists(target_dir):
        return None

    df = pd.read_parquet(target_dir)
    matched_rows = df[df["location_id"] == location_id]

    filtered_data = matched_rows[["value_time", "value"]].copy()
    filtered_data.rename(
        columns={"value_time": "Date", "value": "Discharge"}, inplace=True
    )
    return filtered_data if not filtered_data.empty else None


def plotUSGSStreamflowData(dischargedata, usgs_sites, output_dir, start_date, end_date):
    plt.figure(figsize=(10, 5))
    missing_sites = []

    for usgs_site in usgs_sites:
        data = getUSGSdata(dischargedata, usgs_site, start_date, end_date)
        if data is None:
            missing_sites.append(usgs_site)  # Track missing data
            continue

        # Plot data for valid sites
        plt.plot(
            data["Date"],
            data["Discharge"],
            label=f"USGS streamflow for gauge site: {usgs_site}",
            linewidth=2,
        )

    plt.xlabel("Date (Hourly)", fontsize=14)
    plt.ylabel("Discharge (m³/s)", fontsize=14)
    plt.title("USGS hourly streamflow", fontsize=16)
    plt.legend()
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="-", linewidth=0.3)
    plt.tight_layout()

    # Save directory
    plt_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plt_dir, exist_ok=True)
    plot_dir = os.path.join(plt_dir, "USGSStreamflow.png")
    plt.savefig(plot_dir, dpi=500, bbox_inches="tight")
    plt.show()

    if missing_sites:
        print(
            f"\033[1m****Data not found for the following USGS gauge sites: {', '.join(missing_sites)}****\033[0m"
        )


def plotUSGSStreamflow(huc, usgs_sites, start_date, end_date):
    code_dir, data_dir, output_dir = setup_directories()
    discharge_dir = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "usgs_streamflow"
    )
    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    plotUSGSStreamflowData(discharge_dir, usgs_sites, HUC_dir, start_date, end_date)
