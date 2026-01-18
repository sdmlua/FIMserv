import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from ..plot.nwmfid import getFIDdata
from ..plot.usgs import getUSGSdata

from ..datadownload import setup_directories


# Metrics Calculation
def calculate_metrics(nwm_data, usgs_data):
    r = np.corrcoef(nwm_data, usgs_data)[0, 1]
    beta = np.mean(nwm_data) / np.mean(usgs_data)
    gamma = np.std(nwm_data) / np.std(usgs_data)
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    nse = 1 - np.sum((nwm_data - usgs_data) ** 2) / np.sum(
        (usgs_data - np.mean(usgs_data)) ** 2
    )
    apb = np.sum(np.abs(nwm_data - usgs_data)) / np.sum(usgs_data) * 100

    return {"KGE": kge, "PBias (%)": apb, "NSE": nse}


# Dual-axis visualization
def visualize_comparison(metrics, output_dir, usgs_site, huc):
    metric_names = ["KGE", "NSE", "PBias (%)"]
    metric_values = [
        metrics["KGE"],
        metrics["NSE"],
        metrics["PBias (%)"],
    ]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    bar_colors = ["tab:blue", "tab:purple", "tab:orange"]

    # Bar plot for the first three metrics
    ax1.bar(
        metric_names[:-1],
        metric_values[:-1],
        color=bar_colors[:-1],
        edgecolor="black",
        width=0.7,
        zorder=3,
    )
    ax1.set_ylabel("Metrics Values", color="black", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="black", labelsize=14)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", linestyle="--", alpha=1, zorder=0)

    # Vertical line between the third and fourth bars
    plt.axvline(x=1.5, color="black", linestyle="--", linewidth=1.5)

    # Bar plot for PBias on secondary y-axis
    ax2 = ax1.twinx()
    ax2.bar(
        [metric_names[-1]],
        [metric_values[-1]],
        color=bar_colors[-1],
        edgecolor="black",
        width=0.7,
        zorder=3,
    )
    ax2.set_ylabel("Percentage Bias (%)", color="red", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="red", labelsize=14)
    ax2.set_ylim(0, metric_values[-1] * 1.2)

    padding = 0.01
    for i, val in enumerate(metric_values):
        ax = ax1 if i < 2 else ax2
        text_color = "red" if i == 2 else "black"
        # Adjust padding depending on axis
        pad = padding if i < 2 else val * 0.05  # 5% of PBias value
        ax.text(
            i,
            val + pad,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=text_color,
        )
    # Title and labels
    ax1.set_xlabel("Statistical Metrics", fontsize=16)
    plt.xticks(ticks=range(len(metric_names)), labels=metric_names, fontsize=14)
    plt.tight_layout()

    # Save or display plot
    plt_dir = os.path.join(output_dir, f"flood_{huc}", "Plots")
    os.makedirs(plt_dir, exist_ok=True)
    plot_dir = os.path.join(plt_dir, f"EvaluationMetrics_{usgs_site}.png")
    plt.savefig(plot_dir, dpi=500, bbox_inches="tight")
    plt.show()


def CalculateStatistics(huc, feature_id, usgs_site, start_date, end_date):
    code_dir, data_dir, output_dir = setup_directories()
    discharge_dir_nwm = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "nwm30_retrospective"
    )
    discharge_dir_usgs = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "usgs_streamflow"
    )
    # HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    nwm_data = getFIDdata(discharge_dir_nwm, feature_id, start_date, end_date)
    usgs_data = getUSGSdata(discharge_dir_usgs, usgs_site, start_date, end_date)

    # Ensure dates are datetime objects
    nwm_data["Date"] = pd.to_datetime(nwm_data["Date"])
    usgs_data["Date"] = pd.to_datetime(usgs_data["Date"])

    # Merge datasets
    merged_data = pd.merge(
        nwm_data.rename(columns={"Discharge": "Discharge_NWM"}),
        usgs_data.rename(columns={"Discharge": "Discharge_USGS"}),
        on="Date",
    ).dropna()
    # Calculate metrics
    metrics = calculate_metrics(
        merged_data["Discharge_NWM"], merged_data["Discharge_USGS"]
    )
    print(
        f"***********Metrics for the given USGS gauge {usgs_site} and NWM feature ID {feature_id}***********"
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    # Visualize metrics
    visualize_comparison(metrics, output_dir, usgs_site, huc)
