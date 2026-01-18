# Import Libraries
import os
import numpy as np
import pandas as pd
from pathlib import Path
import s3fs
import xarray
from datetime import datetime, timedelta

from ..datadownload import setup_directories


def get_geoglowsdatafromS3():
    bucket_uri = "s3://geoglows-v2-retrospective/retrospective.zarr"
    region_name = "us-west-2"
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
    s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

    # All data
    ds = xarray.open_zarr(s3store)
    return ds


def get_rivID(hydrotable):
    df = pd.read_csv(hydrotable)
    return df


def getGLOWS_data(
    event_time, hydrotable, data_dir, output_dir, huc, start_date=None, end_date=None
):
    """
    Get GLOWS data for a specific event time and save it to a CSV file.
    """

    value_time = pd.to_datetime(event_time)

    if start_date is None or end_date is None:
        value_time = pd.to_datetime(event_time)

        # Calculate start and end dates
        start_date = value_time - timedelta(days=1)
        tentative_end_date = value_time + timedelta(days=1)
        current_time = datetime.utcnow()
        end_date = min(tentative_end_date, current_time)

    # Get the retrospective dataset
    ds = get_geoglowsdatafromS3()
    hydro_df = pd.read_csv(hydrotable)

    # Map LINKNO to feature_id
    linkno_to_featureid = hydro_df.set_index("LINKNO")["feature_id"].to_dict()
    riv_ids = hydro_df["LINKNO"].tolist()
    filtered_ds = ds["Qout"].sel(rivid=riv_ids).to_dataframe()
    filtered_ds.reset_index(inplace=True)
    filtered_ds["time"] = pd.to_datetime(filtered_ds["time"])
    filtered_df = filtered_ds[
        (filtered_ds["time"] >= start_date) & (filtered_ds["time"] <= end_date)
    ]

    # Map rivid (LINKNO) to feature_id
    filtered_df["feature_id"] = filtered_df["rivid"].map(linkno_to_featureid)
    output_df = filtered_df[["feature_id", "Qout", "time"]]

    output_df.rename(columns={"Qout": "discharge"}, inplace=True)

    # Export the filtered data to a CSV file
    out_dir = Path(output_dir) / "GEOGLOWS"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{huc}_{start_date}_{end_date}_streamflow.csv"
    output_df["feature_id"] = output_df["feature_id"].astype(int)
    output_df.to_csv(output_file, index=False)

    # Filter based on value_time
    value_time_df = output_df[output_df["time"] == value_time]
    value_time_df = value_time_df[["feature_id", "discharge"]]
    value_time_df["feature_id"] = value_time_df["feature_id"].astype(int)

    # Export the value_time data to a separate CSV file
    value_timeSTR = pd.to_datetime(value_time).strftime("%Y%m%d")
    value_time_file = Path(data_dir) / f"GeoGLOWS_{value_timeSTR}_{huc}.csv"
    value_time_df.to_csv(value_time_file, index=False)


# Function to call
def getGEOGLOWSstreamflow(huc, event_time, hydrotable, start_date=None, end_date=None):
    """
    Get GLOWS data for a specific HUC and save it to a CSV file.
    """

    code_dir, data_dir, output_dir = setup_directories()

    HUC_dir = os.path.join(output_dir, f"flood_{huc}")
    # Create a output directory
    getGLOWS_data(event_time, hydrotable, data_dir, HUC_dir, huc, start_date, end_date)
