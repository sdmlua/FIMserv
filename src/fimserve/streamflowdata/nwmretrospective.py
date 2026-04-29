import os
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import teehr.fetching.nwm.retrospective_points as nwm_retro

from ..datadownload import setup_directories


# Aggregated discharge for a certain time range (max, min, mean)
def get_aggregated_discharge(
    retrospective_dir, location_ids_file, start_date, end_date, data_dir, huc, sortby
):
    """
    Calculates max, min, or mean discharge over a specific parquet file range.
    """
    retrospective_dir = Path(retrospective_dir)
    all_data = pd.DataFrame()

    # Load only the relevant parquet file for this range
    formatted_filename = (
        f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
    )
    file_path = retrospective_dir / formatted_filename

    if not file_path.exists():
        print(f"File {file_path} not found for aggregation.")
        return

    df = pd.read_parquet(file_path)

    # Filter by location IDs
    locationID_df = pd.read_csv(location_ids_file)
    target_ids = [f"nwm30-{int(fid)}" for fid in locationID_df["feature_id"]]
    df = df[df["location_id"].isin(target_ids)].copy()
    df["feature_id"] = df["location_id"].str.replace("nwm30-", "")

    # Aggregate based on sortby
    if sortby == "maximum":
        discharge_data = df.groupby("feature_id")["value"].max().reset_index()
    elif sortby == "minimum":
        discharge_data = df.groupby("feature_id")["value"].min().reset_index()
    elif sortby == "mean":
        discharge_data = df.groupby("feature_id")["value"].mean().reset_index()
    else:
        return

    discharge_data.rename(columns={"value": "discharge"}, inplace=True)

    # Save with the requested filename format
    fname = f"NWM_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{sortby}_{huc}.csv"
    output_path = os.path.join(data_dir, fname)
    discharge_data.to_csv(output_path, index=False)
    print(f"Sorted ({sortby}) discharge saved to {output_path}")


def getdischargeforspecifiedtime(
    retrospective_dir, location_ids, specific_date, data_dir, huc, date_type
):
    retrospective_dir = Path(retrospective_dir)
    all_data = pd.DataFrame()

    # Loop through all parquet files in the directory
    for file in retrospective_dir.glob("*.parquet"):
        df = pd.read_parquet(file)
        all_data = pd.concat([all_data, df], ignore_index=True)

    all_data["value_time"] = pd.to_datetime(all_data["value_time"])

    locationID_df = pd.read_csv(location_ids)
    location_ids = [f"nwm30-{int(fid)}" for fid in locationID_df["feature_id"]]

    specific_date = pd.to_datetime(specific_date)
    if date_type == "date":
        filtered_df = all_data[
            (all_data["location_id"].isin(location_ids))
            & (all_data["value_time"].dt.date == specific_date.date())
        ].copy()
        filtered_df["feature_id"] = filtered_df["location_id"].str.replace("nwm30-", "")
        discharge_data = (
            filtered_df.groupby("feature_id")["value"]
            .mean()
            .reset_index()
            .rename(columns={"value": "discharge"})
        )
        formatted_datetime = specific_date.strftime("%Y%m%d")
    else:
        filtered_df = all_data[
            (all_data["location_id"].isin(location_ids))
            & (all_data["value_time"] == specific_date)
        ].copy()
        filtered_df.loc[:, "feature_id"] = filtered_df["location_id"].str.replace(
            "nwm30-", ""
        )
        discharge_data = filtered_df[["feature_id", "value"]].rename(
            columns={"value": "discharge"}
        )
        formatted_datetime = specific_date.strftime("%Y%m%d%H%M%S")

    # Save to a CSV file with the date and HUC as filename
    finalHANDdischarge_dir = os.path.join(
        data_dir, f"NWM_{formatted_datetime}_{huc}.csv"
    )
    if os.path.exists(finalHANDdischarge_dir):
        print(f"Discharge CSV already exists, skipping: {finalHANDdischarge_dir}")
        return
    discharge_data.to_csv(finalHANDdischarge_dir, index=False)
    print(f"Discharge values saved to {finalHANDdischarge_dir}")


def getnwm_discharge(
    start_date,
    end_date,
    fids,
    output_root,
    nwm_version="nwm30",
    variable_name="streamflow",
):
    output_dir = Path(output_root) / "discharge" / f"{nwm_version}_retrospective"
    output_dir.mkdir(parents=True, exist_ok=True)

    formatted_filename = (
        f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
    )
    file_path = output_dir / formatted_filename

    # Check if the file already exists
    if file_path.exists():
        print(
            f"Discharge file already exists in {file_path}, skipping download and getting streamflow for valuetimes"
        )
        return

    location_ids_df = pd.read_csv(fids)
    location_ids = location_ids_df["feature_id"].tolist()

    nwm_retro.nwm_retro_to_parquet(
        nwm_version=nwm_version,
        variable_name=variable_name,
        start_date=start_date,
        end_date=end_date,
        location_ids=location_ids,
        output_parquet_dir=output_dir,
    )
    print(f"NWM discharge data saved to {output_dir}.")


def determinedatatimeformat(date_str):
    if isinstance(date_str, pd.Timestamp):
        return "datetime"
    try:
        datetime.strptime(date_str, "%Y-%m-%d").date()
        return "date"
    except ValueError:
        try:
            datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return "datetime"
        except ValueError:
            return "invalid"


def getNWMretrospectivedata(
    huc=None,
    start_date=None,
    end_date=None,
    value_times=None,
    huc_event_dict=None,
    discharge_sortby=None,
):
    """
    Fetches NWM retrospective discharge data.
    - If huc_event_dict is provided, extracts data for multiple HUCs with specific timestamps.
    - If only huc is provided, fetches data using start_date and end_date or specific value_times.

    :param huc: Single HUC for regular processing.
    :param start_date: Start date for time range data especially for evaluation.
    :param end_date: End date for time range data.
    :param value_times: List of specific timestamps for a single HUC.
    :param huc_event_dict: Dictionary of HUCs with specific timestamps.
    """

    code_dir, data_dir, output_dir = setup_directories()

    # Handle Dictionary Input
    if huc_event_dict:
        for h_id, v_times in huc_event_dict.items():
            _process_huc_request(
                h_id, None, None, v_times, output_dir, data_dir, discharge_sortby
            )

    # Handle Single HUC Input
    else:
        if not huc:
            raise ValueError("You must provide a 'huc'.")
        _process_huc_request(
            huc,
            start_date,
            end_date,
            value_times,
            output_dir,
            data_dir,
            discharge_sortby,
        )


def _process_huc_request(
    huc, start_date, end_date, value_times, output_root, data_dir, discharge_sortby
):
    huc_dir = os.path.join(output_root, f"flood_{huc}")
    if not os.path.exists(huc_dir):
        print(f"Directory for {huc} missing. Run DownloadHUC8 first.")
        return

    fid_path = os.path.join(huc_dir, "feature_IDs.csv")
    discharge_root = os.path.join(huc_dir, "discharge")
    retro_dir = os.path.join(discharge_root, "nwm30_retrospective")
    initial_exists = os.path.exists(retro_dir)

    # Date Range provided
    if start_date and end_date:
        getnwm_discharge(start_date, end_date, fid_path, huc_dir)
        if discharge_sortby:
            get_aggregated_discharge(
                retro_dir,
                fid_path,
                start_date,
                end_date,
                data_dir,
                huc,
                discharge_sortby,
            )

    # Specific Timestamps provided
    if value_times:
        for time in value_times:
            dtype = determinedatatimeformat(time)
            t_obj = pd.to_datetime(time)

            # Determine window
            if dtype == "date":
                lag = (t_obj - timedelta(days=1)).strftime("%Y-%m-%d")
                lead = (t_obj + timedelta(days=1)).strftime("%Y-%m-%d")
            elif dtype == "datetime":
                lag = (t_obj - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
                lead = (t_obj + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            else:
                continue
            getnwm_discharge(lag, lead, fid_path, huc_dir)

            # Extract specific time
            getdischargeforspecifiedtime(
                retro_dir, fid_path, time, data_dir, huc, dtype
            )

            # Cleanup temporary window file if we didn't have retro_dir before
            if not initial_exists:
                tmp_file = os.path.join(
                    retro_dir, f"{lag.replace('-', '')}_{lead.replace('-', '')}.parquet"
                )
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

    # Final cleanup if directory was created just for this session
    if (
        not initial_exists
        and os.path.exists(discharge_root)
        and not (start_date and end_date)
    ):
        shutil.rmtree(discharge_root)
