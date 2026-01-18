import os
import teehr
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from teehr.fetching.usgs.usgs import usgs_to_parquet

from ..datadownload import setup_directories
from ..plot.usgs import getUSGSdata
from ..plot import GetUSGSIDandCorrFID
from .nwmretrospective import determinedatatimeformat


def getusgs_discharge(
    start_date,
    end_date,
    usgs_sites,
    output_root,
):
    output_dir = Path(output_root) / "discharge" / "usgs_streamflow"
    output_dir.mkdir(parents=True, exist_ok=True)

    usgs_to_parquet(
        start_date=start_date,
        end_date=end_date,
        sites=usgs_sites,
        output_parquet_dir=output_dir,
        overwrite_output=True,
    )


# If value_times is mentioned and user need the discharge for specific time
def getdischargeforspecifiedtime(
    retrospective_dir,
    location_ids,
    specific_date,
    data_dir,
    huc,
    date_type,
    feature_ids=None,
):
    retrospective_dir = Path(retrospective_dir)
    all_data = pd.DataFrame()

    for file in retrospective_dir.glob("*.parquet"):
        df = pd.read_parquet(file)
        all_data = pd.concat([all_data, df], ignore_index=True)

    all_data["value_time"] = pd.to_datetime(all_data["value_time"])
    location_ids = [str(lid) for lid in location_ids]
    usgs_formatted_ids = [f"usgs-{lid}" for lid in location_ids]
    specific_date = pd.to_datetime(specific_date)

    # Create mapping from location_id to feature_id
    id_map = {}
    if feature_ids is not None:
        id_map = {f"usgs-{lid}": fid for lid, fid in zip(location_ids, feature_ids)}

    if date_type == "date":
        filtered_df = all_data[
            (all_data["location_id"].isin(usgs_formatted_ids))
            & (all_data["value_time"].dt.date == specific_date.date())
        ].copy()
    else:
        filtered_df = all_data[
            (all_data["location_id"].isin(usgs_formatted_ids))
            & (all_data["value_time"] == specific_date)
        ].copy()

    if feature_ids is not None:
        filtered_df["feature_id"] = filtered_df["location_id"].map(id_map)
    else:
        filtered_df["feature_id"] = filtered_df["location_id"].str.replace(
            "usgs-", "", regex=False
        )

    discharge_data = (
        filtered_df.groupby("feature_id")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "discharge"})
    )
    # Make sure the columns are in int and float
    discharge_data = discharge_data.astype({"feature_id": int, "discharge": float})

    formatted_datetime = (
        specific_date.strftime("%Y%m%d")
        if date_type == "date"
        else specific_date.strftime("%Y%m%d%H%M%S")
    )
    finalHANDdischarge_dir = os.path.join(
        data_dir, f"USGS_{formatted_datetime}_{huc}.csv"
    )
    discharge_data.to_csv(finalHANDdischarge_dir, index=False)
    print(f"Discharge values saved to {finalHANDdischarge_dir}")


def getUSGSsitedata(
    huc=None,
    start_date=None,
    end_date=None,
    usgs_sites=None,
    value_times=None,
    huc_event_dict=None,
):
    """
    If there is no value times, it will just proceed with start and end date and there will be no
    discharge for the particular date or event. If there is a value time, user doesnot need to send the usgs_sites
    it will first get if there is any usgs_sites in the particular HUC and then get the data for the value time and
    assign a feature_id corresponding to the usgs_sites and save in the data/inputs as required by the FIMserv.
    """
    code_dir, data_dir, output_dir = setup_directories()
    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    featureID_dir = os.path.join(HUC_dir, f"feature_IDs.csv")

    def process_value_times(huc_key, value_times_list, allow_cleanup=False):
        site_data = GetUSGSIDandCorrFID(huc_key)
        usgs_ids = site_data["USGS gauge station ID"].tolist()
        feature_ids = site_data["feature_id"].tolist()

        huc_output_dir = os.path.join(output_dir, f"flood_{huc_key}")
        discharge_dir = os.path.join(huc_output_dir, "discharge")
        retrospective_dir = os.path.join(discharge_dir, "usgs_streamflow")

        initial_retrospective_exists = os.path.exists(retrospective_dir)

        for value_time in value_times_list:
            date_type = determinedatatimeformat(value_time)
            value_time = pd.to_datetime(value_time)

            # Time window based on type
            if date_type == "date":
                start = value_time - timedelta(days=1)
                end = min(value_time + timedelta(days=1), datetime.utcnow())
            elif date_type == "datetime":
                start = value_time - timedelta(hours=1)
                end = min(value_time + timedelta(hours=1), datetime.utcnow())
            else:
                print(f"Invalid date format: {value_time}")
                continue

            # Download data
            getusgs_discharge(start, end, usgs_ids, huc_output_dir)

            # Extract specified discharge
            getdischargeforspecifiedtime(
                retrospective_dir,
                usgs_ids,
                value_time,
                data_dir,
                huc_key,
                date_type,
                feature_ids,
            )

        # Clean only if this is part of the huc_event_dict process
        if (
            allow_cleanup
            and not initial_retrospective_exists
            and os.path.exists(discharge_dir)
        ):
            shutil.rmtree(discharge_dir)

    # Multiple HUCs from dictionary
    if huc_event_dict is not None:
        for huc_key, vtimes in huc_event_dict.items():
            process_value_times(huc_key, vtimes, allow_cleanup=True)
        return

    # Single HUC, with event times
    if value_times is not None:
        if huc is None:
            raise ValueError("HUC must be provided when using value_times.")

        # Range-based download first
        if start_date and end_date:
            output_directory = os.path.join(output_dir, f"flood_{huc}")
            if usgs_sites is None:
                usgs_sites = GetUSGSIDandCorrFID(huc)["USGS gauge station ID"].tolist()
            getusgs_discharge(start_date, end_date, usgs_sites, output_directory)

        # process value times
        process_value_times(huc, value_times)
        return

    # Date range only, optional HUC and USGS sites
    output_directory = (
        os.getcwd() if huc is None else os.path.join(output_dir, f"flood_{huc}")
    )
    if usgs_sites is None and huc is not None:
        usgs_sites = GetUSGSIDandCorrFID(huc)["USGS gauge station ID"].tolist()

    getusgs_discharge(start_date, end_date, usgs_sites, output_directory)
