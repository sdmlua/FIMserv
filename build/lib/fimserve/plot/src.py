import os
import pandas as pd
import matplotlib.pyplot as plt

from ..datadownload import setup_directories


def filterhydroID(file_path, hydro_ids, branch_ids):
    df = pd.read_csv(file_path)

    df["HydroID"] = df["HydroID"].astype(str)
    df["branch_id"] = df["branch_id"].astype(str)

    filtered_dfs = []
    for hydro_id, branch_id in zip(hydro_ids, branch_ids):
        filtered_df = df[(df["HydroID"] == hydro_id) & (df["branch_id"] == branch_id)]
        result_df = filtered_df[["stage", "default_discharge_cms"]]
        filtered_dfs.append(result_df)
    return filtered_dfs


def plotsrc(file, hydro_ids, branch_ids, output_dir, discharge_value=None):
    data_list = filterhydroID(file, hydro_ids, branch_ids)
    plt.figure(figsize=(5, 4))
    cmap = plt.get_cmap("tab10" if len(data_list) <= 10 else "hsv")
    colors = [cmap(i / len(data_list)) for i in range(len(data_list))]

    # Fill colors for the markers
    marker_fill_colors = ["pink", "lightblue", "lightgreen", "lightcoral", "yellow"]

    for i, data in enumerate(data_list):
        plt.plot(
            data["default_discharge_cms"],
            data["stage"],
            label=f"Synthetic rating curve for HydroID: {hydro_ids[i]}",
            linestyle="-",
            linewidth=2.5,
            color=colors[i],
        )

        if discharge_value:
            stage_at_discharge = data["stage"].iloc[
                (data["default_discharge_cms"] - discharge_value).abs().argmin()
            ]

            plt.plot(
                [discharge_value, discharge_value],
                [0, stage_at_discharge],
                linestyle="--",
                color=colors[i],
                linewidth=2,
            )
            plt.plot(
                [0, discharge_value],
                [stage_at_discharge, stage_at_discharge],
                linestyle="--",
                color=colors[i],
                linewidth=2,
            )
            # Marker at the intersection
            plt.scatter(
                discharge_value,
                stage_at_discharge,
                color=marker_fill_colors[i % len(marker_fill_colors)],
                edgecolor="black",
                s=200,
                marker="*",
                zorder=5,
                label=f"For discharge {discharge_value} cms, stage: {stage_at_discharge:.2f} m",
            )

    plt.xlabel("Discharge (m³/s)", fontsize=14)
    plt.ylabel("Stage (in m)", fontsize=14)
    plt.title("Synthetic Rating Curves", fontsize=16)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(True, which="both", linestyle="-", linewidth=0.3)

    # Save dir
    plt_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plt_dir, exist_ok=True)
    plot_dir = os.path.join(plt_dir, f"SRC_{hydro_ids[0]}.png")
    plt.savefig(plot_dir, dpi=500, bbox_inches="tight")
    plt.show()


def plotSRC(huc, hydro_ids, branch_ids, discharge_value=None):
    code_dir, data_dir, output_dir = setup_directories()
    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    hydrotable_dir = os.path.join(output_dir, f"flood_{huc}", huc, "hydrotable.csv")
    plotsrc(hydrotable_dir, hydro_ids, branch_ids, HUC_dir, discharge_value)
