import os
import matplotlib.pyplot as plt

from .nwmfid import getFIDdata
from .usgs import getUSGSdata
from ..datadownload import setup_directories


def plotcomparision(
    data_dir_nwm, data_dir_usgs, feature_id, usgs_site, output_dir, start_date, end_date
):
    nwm_data = getFIDdata(data_dir_nwm, feature_id, start_date, end_date)
    usgs_data = getUSGSdata(data_dir_usgs, usgs_site, start_date, end_date)

    plt.figure(figsize=(10, 5))
    # Plot NWM data with solid line
    plt.plot(
        nwm_data["Date"],
        nwm_data["Discharge"],
        label=f"NWM Streamflow of feature ID {feature_id}",
        linestyle="solid",
        color="#167693",
        linewidth=2,
    )

    # Plot USGS data with dashed line
    plt.plot(
        usgs_data["Date"],
        usgs_data["Discharge"],
        label=f"USGS Streamflow of gauged site {usgs_site}",
        linestyle="dashed",
        color="#BF4037",
        linewidth=2,
    )

    plt.xlabel("Date (Hourly)", fontsize=14)
    plt.ylabel("Discharge (m³/s)", fontsize=14)
    plt.title(f"Discharge comparison between USGS and NWM streamflow", fontsize=16)
    plt.legend()
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # Save dir
    plt_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plt_dir, exist_ok=True)
    plot_dir = os.path.join(plt_dir, f"NWMvsUSGS_{usgs_site}.png")
    plt.grid(True, which="both", linestyle="-", linewidth=0.3)
    plt.savefig(plot_dir, dpi=500, bbox_inches="tight")
    plt.show()


def CompareNWMnUSGSStreamflow(huc, feature_id, usgs_site, start_date, end_date):
    code_dir, data_dir, output_dir = setup_directories()
    discharge_dir_nwm = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "nwm30_retrospective"
    )
    discharge_dir_usgs = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "usgs_streamflow"
    )
    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    plotcomparision(
        discharge_dir_nwm,
        discharge_dir_usgs,
        feature_id,
        usgs_site,
        HUC_dir,
        start_date,
        end_date,
    )
