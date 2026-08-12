"""
Author: Supath Dhital (sdhital@crimson.ua.edu)
Date Updated: August 12, 2026

Downloads NWM Analysis and Assimilation (AnA) discharge, the model's gauge assimilated
best estimate of past conditions, as a continuous hourly series for a HUC8.
"""

import os
import pandas as pd
from pathlib import Path
import netCDF4 as nc
from datetime import datetime, timedelta

from ..datadownload import setup_directories
from .forecasteddata import download_public_file, _rmtree

URL_BASE = "https://storage.googleapis.com/national-water-model"

# First AnA cycle published on the NWM bucket; older events need the retrospective
ARCHIVE_START = datetime(2018, 9, 17)

DATETIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d"]


def isdateonly(value):
    """True when the user gave a plain day (no hour), so the whole day is meant."""
    if isinstance(value, (datetime, pd.Timestamp)):
        return False
    return len(str(value).strip()) <= 10


def parsedatetime(value, end_of_day=False):
    """Parse a date or datetime to the hour. A plain day snaps to 00UTC,
    or to 23UTC when end_of_day is set."""
    if isinstance(value, (datetime, pd.Timestamp)):
        return (
            pd.Timestamp(value)
            .to_pydatetime()
            .replace(minute=0, second=0, microsecond=0)
        )

    for fmt in DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d" and end_of_day:
            return parsed.replace(hour=23)
        return parsed.replace(minute=0, second=0, microsecond=0)

    raise ValueError(
        f"Unrecognized date '{value}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
    )


def hourlyrange(start, end):
    hours, current = [], start
    while current <= end:
        hours.append(current)
        current += timedelta(hours=1)
    return hours


# AnA counts backwards from its cycle, so tm00 of cycle tHHz is valid exactly at HH UTC
def getanahour(valid_time, feature_ids, staging_dir):
    """Download the tm00 AnA file valid at valid_time and return the HUC discharge."""
    filename = f"nwm.t{valid_time.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc"
    url = f"{URL_BASE}/nwm.{valid_time:%Y%m%d}/analysis_assim/{filename}"
    file_path = os.path.join(staging_dir, filename)

    try:
        download_public_file(url, file_path)
    except Exception as e:
        print(f"Failed to download AnA for {valid_time:%Y-%m-%d %H}UTC: {e}")
        return None

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"No AnA data available for {valid_time:%Y-%m-%d %H}UTC, skipping.")
        return None

    try:
        ds = nc.Dataset(file_path, "r")
        data_df = pd.DataFrame(
            {
                "feature_id": ds.variables["feature_id"][:],
                "discharge": ds.variables["streamflow"][:],
            }
        )
        ds.close()
    except Exception as e:
        print(f"Error reading NetCDF file {file_path}: {e}")
        return None
    finally:
        # Each CONUS file is ~11 MB, so drop it as soon as the HUC values are read
        if os.path.exists(file_path):
            os.remove(file_path)

    data_df = data_df[data_df["feature_id"].isin(feature_ids)].copy()
    data_df["value_time"] = valid_time
    return data_df


def getanadischarge(start, end, feature_ids, ana_dir, huc):
    """Builds the hourly discharge parquet for the range, reusing it when present."""
    parquet_path = os.path.join(
        ana_dir, f"NWM_data_assim_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    )
    if os.path.exists(parquet_path):
        print(f"Discharge file already exists in {parquet_path}, skipping download.")
        return parquet_path

    staging_dir = os.path.join(ana_dir, "netCDF")
    hourly_dir = os.path.join(ana_dir, "hourly")
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(hourly_dir, exist_ok=True)

    hours = hourlyrange(start, end)
    print(f"Downloading {len(hours)} hourly AnA timestep(s) for HUC {huc}...")

    # Each hour is filtered to the HUC and written out before the next one starts, so
    # an interrupted run resumes instead of downloading the whole range again
    for count, valid_time in enumerate(hours, 1):
        hourly_path = os.path.join(hourly_dir, f"{valid_time:%Y%m%d_%H}.parquet")
        if not os.path.exists(hourly_path):
            discharge_data = getanahour(valid_time, feature_ids, staging_dir)
            if discharge_data is not None and not discharge_data.empty:
                discharge_data.to_parquet(hourly_path, index=False)
        if count % 24 == 0 or count == len(hours):
            print(f"  {count}/{len(hours)} done ({valid_time:%Y-%m-%d %H}UTC)")

    _rmtree(staging_dir)

    hourly_files = sorted(Path(hourly_dir).glob("*.parquet"))
    if not hourly_files:
        _rmtree(hourly_dir)
        print("No AnA discharge retrieved for the requested period.")
        return None

    all_data = pd.concat(
        [pd.read_parquet(file) for file in hourly_files], ignore_index=True
    )
    all_data = all_data[["feature_id", "value_time", "discharge"]]
    all_data.to_parquet(parquet_path, index=False)
    _rmtree(hourly_dir)
    print(f"AnA discharge data saved to {parquet_path}.")
    return parquet_path


def aggregatedischarge(discharge_data, sort_by):
    """Collapse the hourly timesteps into one discharge value per feature_id."""
    grouped = discharge_data.groupby("feature_id")["discharge"]

    if sort_by == "minimum":
        aggregated = grouped.min()
    elif sort_by == "median":
        aggregated = grouped.median()
    elif sort_by == "mean":
        aggregated = grouped.mean()
    else:
        aggregated = grouped.max()

    return aggregated.reset_index()


def savedischarge(discharge_data, data_dir, filename):
    output_path = os.path.join(data_dir, filename)
    discharge_data.to_csv(output_path, index=False)
    print(f"Discharge values saved to {output_path}")


def getdischargeforspecifiedtime(all_data, valuetime, wholeday, sort_by, data_dir, huc):
    """Extracts one value time from the parquet, a whole day or a single hour."""
    if wholeday:
        selected = all_data[all_data["value_time"].dt.date == valuetime.date()]
        filename = f"analysis_assim_{huc}_{valuetime:%Y%m%d}_{sort_by}.csv"
    else:
        selected = all_data[all_data["value_time"] == pd.Timestamp(valuetime)]
        filename = f"analysis_assim_{huc}_{valuetime:%Y%m%d}_{valuetime:%H}UTC.csv"

    if selected.empty:
        print(f"No AnA discharge found for {valuetime:%Y-%m-%d %H}UTC.")
        return

    if wholeday:
        discharge_data = aggregatedischarge(selected, sort_by)
    else:
        discharge_data = selected[["feature_id", "discharge"]]

    savedischarge(discharge_data, data_dir, filename)


def getNWManalysisAssim(
    huc,
    start_date,
    end_date,
    value_times=None,
    continuous_discharge=False,
    sort_by="maximum",
):
    """
    Fetches NWM Analysis and Assimilation (AnA) discharge for a HUC8.

    AnA is the model's best estimate of what already happened, so it is indexed by valid
    time instead of a forecast cycle. One value per reach is taken from the tm00 file of
    each hourly cycle. The whole range is stored as a single parquet under the HUC
    discharge folder, and the requested timesteps are written as CSVs to the inputs.

    :param huc: HUC8 ID.
    :param start_date: Range start as 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
    :param end_date: Range end; a day without an hour runs through 23UTC.
    :param value_times: One timestamp, or a list of them, inside the range, matching
        getNWMretrospectivedata. A timestamp with an hour saves that exact hour; a plain
        day saves the whole day aggregated by sort_by.
    :param continuous_discharge: Save one CSV per hour across the whole range.
    :param sort_by: Aggregation for a plain day: maximum, minimum, median or mean.
    """
    code_dir, data_dir, output_dir = setup_directories()

    huc_dir = os.path.join(output_dir, f"flood_{huc}")
    featureIDs = Path(huc_dir, "feature_IDs.csv")
    if not featureIDs.exists():
        print(f"Directory for {huc} missing. Run DownloadHUC8 first.")
        return

    start = parsedatetime(start_date)
    end = parsedatetime(end_date, end_of_day=True)
    if end < start:
        raise ValueError("end_date must be on or after start_date.")

    if end < ARCHIVE_START:
        print(
            f"NWM AnA starts {ARCHIVE_START:%Y-%m-%d}. For earlier events use "
            "getNWMretrospectivedata instead."
        )
        return
    if start < ARCHIVE_START:
        print(
            f"NWM AnA is unavailable before {ARCHIVE_START:%Y-%m-%d}, "
            "starting the download from that date."
        )
        start = ARCHIVE_START

    # Validate every value time up front so nothing downloads before inputs are checked
    requested_times = []
    if value_times:
        entries = (
            value_times if isinstance(value_times, (list, tuple)) else [value_times]
        )
        for entry in entries:
            valuetime = parsedatetime(entry)
            if not start <= valuetime <= end:
                raise ValueError(
                    f"value_times entry '{entry}' must fall within the date range."
                )
            requested_times.append((valuetime, isdateonly(entry)))

    feature_ids = pd.read_csv(featureIDs)["feature_id"]
    ana_dir = os.path.join(huc_dir, "discharge", "analysis_assim")
    os.makedirs(ana_dir, exist_ok=True)

    # The whole range is always cached as one parquet, the CSVs are extracted from it
    parquet_path = getanadischarge(start, end, feature_ids, ana_dir, huc)
    if parquet_path is None:
        return

    if not requested_times and not continuous_discharge:
        return

    all_data = pd.read_parquet(parquet_path)
    all_data["value_time"] = pd.to_datetime(all_data["value_time"])

    for valuetime, wholeday in requested_times:
        getdischargeforspecifiedtime(
            all_data, valuetime, wholeday, sort_by, data_dir, huc
        )

    if continuous_discharge:
        for valuetime, group in all_data.groupby("value_time"):
            savedischarge(
                group[["feature_id", "discharge"]],
                data_dir,
                f"analysis_assim_{huc}_{valuetime:%Y%m%d}_{valuetime:%H}UTC.csv",
            )
