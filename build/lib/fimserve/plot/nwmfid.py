import pandas as pd
import os
import matplotlib.pyplot as plt

from ..datadownload import setup_directories


def getFIDdata(data_dir, feature_id, start_date, end_date):
    location_id = f"nwm30-{feature_id}"

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    start_dateSTR = start_date.strftime("%Y%m%d")
    end_dateSTR = end_date.strftime("%Y%m%d")
    target_file = f"{start_dateSTR}_{end_dateSTR}.parquet"

    target_dir = os.path.join(data_dir, target_file)

    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            f"No NWM data found for the date range: {target_file} in {data_dir}. "
            "Please check the date range that you downloaded for that HUC."
        )

    df = pd.read_parquet(target_dir)
    matched_rows = df[df["location_id"] == location_id]

    # Prepare the filtered data with desired columns
    filtered_data = matched_rows[["value_time", "value"]].copy()
    filtered_data.rename(
        columns={"value_time": "Date", "value": "Discharge"}, inplace=True
    )

    return filtered_data


# For the default feature_id
def getFeatureWithMaxDischarge(data_dir, start_date, end_date):
    start_dateSTR = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_dateSTR = pd.to_datetime(end_date).strftime("%Y%m%d")
    target_file = f"{start_dateSTR}_{end_dateSTR}.parquet"

    target_dir = os.path.join(data_dir, target_file)

    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            f"No NWM data found for the date range: {target_file} in {data_dir}. "
            "Please check the date range that you downloaded for that HUC."
        )

    df = pd.read_parquet(target_dir)
    max_discharge_row = df.loc[df["value"].idxmax()]
    max_feature_id = max_discharge_row["location_id"].split("-")[1]
    return max_feature_id


def plotNWMStreamflowData(dischargedata, feature_ids, output_dir, start_date, end_date):
    plt.figure(figsize=(10, 5))
    missing_ids = []
    plotted_ids = []

    for feature_id in feature_ids:
        try:
            data = getFIDdata(dischargedata, feature_id, start_date, end_date)
            if data.empty:
                raise ValueError(f"No data for feature ID: {feature_id}")

            # Plot data only if it exists
            plt.plot(
                data["Date"],
                data["Discharge"],
                label=f"NWM streamflow for feature ID: {feature_id}",
                linewidth=2,
            )
            plotted_ids.append(feature_id)
        except (FileNotFoundError, ValueError):
            missing_ids.append(feature_id)

    plt.xlabel("Date (Hourly)", fontsize=14)
    plt.ylabel("Discharge (m³/s)", fontsize=14)
    plt.title("NWM hourly streamflow", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="-", linewidth=0.3)
    plt.tight_layout()

    if plotted_ids:
        plt.legend()
        # Save the plot
        plt_dir = os.path.join(output_dir, "Plots")
        os.makedirs(plt_dir, exist_ok=True)
        plot_dir = os.path.join(plt_dir, "NWMStreamflow.png")
        plt.savefig(plot_dir, dpi=500, bbox_inches="tight")
        plt.show()
    else:
        print(
            "\033[1m****No valid data found for any provided feature IDs. Plot not generated.****\033[0m"
        )

    if missing_ids:
        print(
            f"\033[1m****Data not found for the following NWM feature IDs: {', '.join(map(str, missing_ids))}****\033[0m"
        )


# Main function to drive the process
def plotNWMStreamflow(huc, start_date, end_date, feature_ids=None):
    code_dir, data_dir, output_dir = setup_directories()
    huc_dir = os.path.join(output_dir, f"flood_{huc}")
    discharge_dir = os.path.join(
        output_dir, f"flood_{huc}", "discharge", "nwm30_retrospective"
    )
    if feature_ids is None or not feature_ids:
        max_feature_id = getFeatureWithMaxDischarge(discharge_dir, start_date, end_date)
        print(
            f"*****No feature_id provided. Using the feature with max discharge: {max_feature_id}******"
        )
        feature_ids = [max_feature_id]

    plotNWMStreamflowData(discharge_dir, feature_ids, huc_dir, start_date, end_date)
