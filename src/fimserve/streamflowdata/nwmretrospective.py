import os
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import teehr.fetching.nwm.retrospective_points as nwm_retro

from ..datadownload import setup_directories


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
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return "date"
    except ValueError:
        try:
            parsed_datetime = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return "datetime"
        except ValueError:
            return "invalid"


def getNWMretrospectivedata(
    huc=None, start_date=None, end_date=None, value_times=None, huc_event_dict=None
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

    if huc_event_dict:
        for huc, value_times in huc_event_dict.items():
            HUC_dir = os.path.join(output_dir, f"flood_{huc}")
            # If huc directory does not exist, print the message and continue to the next HUC
            if not os.path.exists(HUC_dir):
                print(
                    f"Directory for {huc} does not exist. Download it with DownloadHUC8 module."
                )
                continue
            featureID_dir = os.path.join(HUC_dir, f"feature_IDs.csv")
            discharge_dir = os.path.join(HUC_dir, "discharge")
            retrospective_dir = os.path.join(discharge_dir, "nwm30_retrospective")

            if not os.path.exists(featureID_dir):
                continue

            initial_retrospective = os.path.exists(retrospective_dir)

            for time in value_times:
                date_type = determinedatatimeformat(time)
                time_obj = pd.to_datetime(time)
                if date_type == "date":
                    lag_date = (time_obj - timedelta(days=1)).strftime("%Y-%m-%d")
                    lead_date = (time_obj + timedelta(days=1)).strftime("%Y-%m-%d")
                elif date_type == "datetime":
                    lag_date = (time_obj - timedelta(hours=1)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    lead_date = (time_obj + timedelta(hours=1)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                else:
                    print(f"Invalid date format: {time}")
                    continue

                getnwm_discharge(lag_date, lead_date, featureID_dir, HUC_dir)

                # Extract discharge values for the actual timestamp
                getdischargeforspecifiedtime(
                    retrospective_dir, featureID_dir, time, data_dir, huc, date_type
                )

                formatted_filename = (
                    f"{lag_date.replace('-', '')}_{lead_date.replace('-', '')}.parquet"
                )
                file_path = os.path.join(retrospective_dir, formatted_filename)
                # Delete the newly created file
                if os.path.exists(file_path):
                    os.remove(file_path)

            # If the discharge directory is not exist just delete it completely once operation is done
            if not initial_retrospective and os.path.exists(discharge_dir):
                shutil.rmtree(discharge_dir)
            print(f"Processing complete for {time} of {huc}.")

    else:
        if not huc:
            raise ValueError("You must provide a valid 'huc'.")
        if not ((start_date and end_date) or value_times):
            raise ValueError(
                "You must provide either a date range (start_date and end_date) or a list of value_times."
            )

        HUC_dir = os.path.join(output_dir, f"flood_{huc}")
        featureID_dir = os.path.join(HUC_dir, "feature_IDs.csv")

        if start_date and end_date:
            start_date_obj = pd.to_datetime(start_date)
            end_date_obj = pd.to_datetime(end_date)

            if (
                start_date_obj == end_date_obj
                and determinedatatimeformat(start_date) == "date"
            ):
                print(
                    "Please provide a start and end date at least 1 day apart or mention hours."
                )
                return
            elif start_date_obj > end_date_obj:
                print("Start date cannot be after the end date.")
                return

            getnwm_discharge(start_date, end_date, featureID_dir, HUC_dir)

        if value_times:
            retrospective_dir = os.path.join(
                HUC_dir, "discharge", "nwm30_retrospective"
            )
            for time in value_times:
                datetype = determinedatatimeformat(time)
                getdischargeforspecifiedtime(
                    retrospective_dir, featureID_dir, time, data_dir, huc, datetype
                )
