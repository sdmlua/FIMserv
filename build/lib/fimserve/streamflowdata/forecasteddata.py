import os
import re
import shutil
import requests
import pandas as pd
from pathlib import Path
import netCDF4 as nc
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup

from ..datadownload import setup_directories


def adjust_hour(hour, forecast_range):
    hour = int(hour)
    if forecast_range == "shortrange":
        return min(max(hour, 0), 23)
    elif forecast_range == "mediumrange":
        valid_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    elif forecast_range == "longrange":
        valid_hours = [0, 6, 12, 18]
    else:
        return hour
    adjusted_hour = max([h for h in valid_hours if h <= hour] or [valid_hours[0]])
    return adjusted_hour


# Delete the forecast directory if it is empty
import os, shutil, stat, time


def _rmtree(path: str, retries: int = 3, delay: float = 0.2) -> None:
    """Robust rmtree: clears read-only bits and retries briefly."""

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
        except Exception:
            pass
        try:
            func(p)
        except Exception:
            pass

    for i in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, onerror=_onerror)
            return
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(delay)


def cleanup_download_tree(
    download_dir: str, final_date_output_dir: str, csv_tmp_dir: str
) -> None:
    """Remove staging CSVs, the dated netCDF dir, the netCDF root, and the forecast folder."""
    _rmtree(csv_tmp_dir)
    _rmtree(final_date_output_dir)
    _rmtree(os.path.join(download_dir, "netCDF"))
    _rmtree(download_dir)


def download_public_file(url, destination_path):
    try:
        response = requests.get(url)
        if response.status_code == 404:
            return
        response.raise_for_status()
        with open(destination_path, "wb") as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        raise e


def download_nc_files(date_str, current_hour, download_dir, url_base, forecast_range):
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    start_date = datetime(2018, 9, 17)
    end_date = datetime(2019, 6, 18)

    # Adjust forecast_type based on the date range
    if start_date <= date_obj <= end_date:
        forecast_type = re.sub(
            r"(?i)mediumrange|medium[-\s]?range", "medium_range", forecast_range
        )
    else:
        forecast_type = re.sub(
            r"(?i)mediumrange|medium[-\s]?range", "medium_range_mem1", forecast_range
        )

    forecast_type = re.sub(
        r"(?i)shortrange|short[-\s]?range", "short_range", forecast_type
    )
    forecast_type = re.sub(
        r"(?i)longrange|long[-\s]?range", "long_range_mem1", forecast_type
    )

    url = f"{url_base}/nwm.{date_str}/{forecast_type}/"

    date_output_dir = os.path.join(download_dir, "netCDF", date_str)
    os.makedirs(date_output_dir, exist_ok=True)

    # Possible File patterns for each forecast type
    expected_forecast_files = []
    if forecast_type == "short_range":
        expected_forecast_files = [
            f"nwm.t{current_hour:02d}z.short_range.channel_rt.f{hour:03d}.conus.nc"
            for hour in range(1, 18)
        ]
    elif forecast_type == "medium_range":
        expected_forecast_files = [
            f"nwm.t{current_hour:02d}z.medium_range.channel_rt.f{hour:03d}.conus.nc"
            for hour in range(3, 240, 3)
        ]
    elif forecast_type == "medium_range_mem1":
        expected_forecast_files = [
            f"nwm.t{current_hour:02d}z.medium_range.channel_rt_1.f{hour:03d}.conus.nc"
            for hour in range(3, 240, 3)
        ]
    elif forecast_type == "long_range_mem1":
        expected_forecast_files = [
            f"nwm.t{current_hour:02d}z.long_range.channel_rt_1.f{hour:03d}.conus.nc"
            for hour in range(6, 720, 6)
        ]

    successful_downloads = []

    for forecast_file in expected_forecast_files:  # Use the newly defined list
        file_url = os.path.join(url, forecast_file)
        file_path = os.path.join(date_output_dir, forecast_file)

        try:
            download_public_file(file_url, file_path)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                successful_downloads.append(forecast_file)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {forecast_file}: {e}")

    # Return True only if ALL expected files were successfully downloaded
    if (
        len(successful_downloads) == len(expected_forecast_files)
        and len(expected_forecast_files) > 0
    ):
        return True, date_output_dir
    else:
        for downloaded_file in os.listdir(date_output_dir):
            os.remove(os.path.join(date_output_dir, downloaded_file))
        os.rmdir(date_output_dir)
        return False, None


# Process netcf and get the file in CSV format and extract all feature_is's discharge data
def processnetCDF(netcdf_file_path, filter_df, output_folder_path):
    base_filename = os.path.basename(netcdf_file_path).replace(".nc", "")
    output_csv_file_path = os.path.join(output_folder_path, f"{base_filename}.csv")

    try:
        ds = nc.Dataset(netcdf_file_path, "r")
        streamflow_data = ds.variables["streamflow"][:]
        feature_ids = ds.variables["feature_id"][:]
        ds.close()
    except Exception as e:
        print(f"Error reading NetCDF file {netcdf_file_path}: {e}")
        return

    if len(streamflow_data) == 0 or len(feature_ids) == 0:
        print(f"No data found in {netcdf_file_path}")
        return

    data_df = pd.DataFrame({"feature_id": feature_ids, "discharge": streamflow_data})

    filtered_df = data_df[data_df["feature_id"].isin(filter_df["feature_id"])]
    merged_df = pd.merge(filter_df[["feature_id"]], filtered_df, on="feature_id")
    merged_df.to_csv(output_csv_file_path, index=False)


def ProcessForecasts(
    CSVFILES, forecast_date, hour, forecast_range, sort_by, data_dir, huc
):
    merge_folder = os.path.join(CSVFILES, "mergedAndSorted")
    os.makedirs(merge_folder, exist_ok=True)

    # Get all CSV files in the output folder
    if forecast_range == "longrange":
        csv_files = sorted(
            [
                file
                for file in os.listdir(CSVFILES)
                if file.endswith(".csv")
                and file.startswith(f"nwm.t{hour:02d}z.long_range")
            ]
        )
    elif forecast_range == "mediumrange":
        csv_files = sorted(
            [
                file
                for file in os.listdir(CSVFILES)
                if file.endswith(".csv")
                and file.startswith(f"nwm.t{hour:02d}z.medium_range")
            ]
        )

    elif forecast_range == "shortrange":
        csv_files = sorted(
            [
                file
                for file in os.listdir(CSVFILES)
                if file.endswith(".csv")
                and file.startswith(f"nwm.t{hour:02d}z.short_range")
            ]
        )
        # Process each file for shortrange
        pattern = re.compile(r"\.f(\d{3})\.")
        current_date = datetime.strptime(forecast_date, "%Y%m%d")
        base_hour = hour
        day_shift = 0

        for csv_file in csv_files:
            match = pattern.search(csv_file)
            if match:
                forecast_hour = int(match.group(1))
                adjusted_forecast_hour = base_hour + forecast_hour
                if adjusted_forecast_hour >= 24:
                    adjusted_forecast_hour %= 24
                    if adjusted_forecast_hour == 0:
                        day_shift += 1
                adjusted_date = (current_date + timedelta(days=day_shift)).strftime(
                    "%Y%m%d"
                )

                sorted_file_name = f"{forecast_range}_{huc}_{adjusted_date}_{adjusted_forecast_hour:02d}UTC.csv"
                sorted_file_path = os.path.join(data_dir, sorted_file_name)
                original_file_path = os.path.join(CSVFILES, csv_file)
                os.rename(original_file_path, sorted_file_path)
        return

    # Calculating the day offset based on the forecast hour
    pattern = re.compile(r"\.f(\d{3})\.")
    current_date = datetime.strptime(forecast_date, "%Y%m%d")
    daily_groups = {}

    for csv_file in csv_files:
        match = pattern.search(csv_file)
        if match:
            forecast_hour = int(match.group(1))
            day_offset = (forecast_hour + hour) // 24
            group_date = (current_date + timedelta(days=day_offset)).strftime("%Y%m%d")

            if group_date not in daily_groups:
                daily_groups[group_date] = []
            daily_groups[group_date].append(csv_file)
        else:
            print(f"Filename does not match expected pattern: {csv_file}")

    # Merge and sort files for each day
    for group_date, group_files in daily_groups.items():
        combined_df = pd.concat(
            [pd.read_csv(os.path.join(CSVFILES, file)) for file in group_files]
        )
        if sort_by == "minimum":
            sorted_df = (
                combined_df.groupby("feature_id")["discharge"].min().reset_index()
            )
        elif sort_by == "median":
            sorted_df = (
                combined_df.groupby("feature_id")["discharge"].median().reset_index()
            )
        else:
            sorted_df = (
                combined_df.groupby("feature_id")["discharge"].max().reset_index()
            )

        sorted_file_name = f"{hour:02d}UTC_{forecast_range}_{group_date}_{huc}.csv"
        sorted_file_path = os.path.join(data_dir, sorted_file_name)
        sorted_df.to_csv(sorted_file_path, index=False)


def main(
    download_dir,
    output_csv_filename,
    HUC,
    data_dir,
    output_dir,
    forecast_range,
    forecast_date=None,
    hour=None,
    sort_by="maximum",
    url_base="https://storage.googleapis.com/national-water-model",
):
    if forecast_date:
        date_obj = datetime.strptime(forecast_date, "%Y-%m-%d")
        forecast_date = date_obj.strftime("%Y%m%d")
    else:
        forecast_date = datetime.utcnow().strftime("%Y%m%d")

    """ This block auto-selects the latest available forecast hour if not provided.
    For shortrange, it checks if the current UTC minute is before 45—if so, it uses the previous hour; otherwise, 
    it uses the current hour. For other forecast types, it picks the latest valid issue time (e.g., 00, 06, 12, 18 UTC) using adjust_hour()"""
    current_download_date = (
        forecast_date
        if forecast_date
        else datetime.now(timezone.utc).strftime("%Y%m%d")
    )
    initial_utc_now = datetime.now(timezone.utc)

    if hour is None:
        if initial_utc_now.minute < 45 and forecast_range == "shortrange":
            current_download_hour = adjust_hour(
                initial_utc_now.hour - 1, forecast_range
            )
        else:
            current_download_hour = adjust_hour(initial_utc_now.hour, forecast_range)
    else:
        original_hour = hour
        current_download_hour = adjust_hour(hour, forecast_range)
        if current_download_hour != original_hour:
            print(
                f"Adjusted forecast hour from {original_hour} to {current_download_hour} for {forecast_range} as per the forecast range and data availability rules."
            )
    print(f"Starting download attempts...")

    """Retry downloading forecast files by decrementing the forecast hour based on forecast range:
    - shortrange: tries every past hour (up to 24 hours)
    - mediumrange: tries every past 3 hours (up to 48 hours)
    - longrange: tries every past 6 hours (up to 48 hours)
    If hour wraps around to previous day, the forecast date is decremented accordingly."""
    success = False
    if forecast_range == "shortrange":
        decrement = 1
        max_attempts = 24  # 24 hours
    elif forecast_range == "mediumrange":
        decrement = 3
        max_attempts = 16  # 2 days (16 × 3 = 48 hrs)
    elif forecast_range == "longrange":
        decrement = 6
        max_attempts = 8  # 2 days (8 × 6 = 48 hrs)
    else:
        decrement = 1
        max_attempts = 24

    attempts = 0
    final_date_output_dir = None  # Initialize to None

    while not success and attempts < max_attempts:
        print(
            f"Attempt {attempts + 1}/{max_attempts}: Trying date {current_download_date}, hour {current_download_hour:02d}Z for {forecast_range}..."
        )

        success, retrieved_date_output_dir = download_nc_files(
            current_download_date,
            current_download_hour,
            download_dir,
            url_base,
            forecast_range,
        )

        if success:
            final_date_output_dir = retrieved_date_output_dir
            print(
                f"Successfully downloaded a complete set for {current_download_date} at {current_download_hour:02d}Z."
            )
            break
        else:
            prev_hour_for_log = current_download_hour
            current_download_hour = (current_download_hour - decrement) % 24
            if current_download_hour > prev_hour_for_log:
                date_obj = datetime.strptime(
                    current_download_date, "%Y%m%d"
                ) - timedelta(days=1)
                current_download_date = date_obj.strftime("%Y%m%d")
            attempts += 1
            # Updated print statement here
            print(
                f"Download failed for {current_download_date} at {prev_hour_for_log:02d}Z. No complete data found for this period. Retrying by going back {decrement} hour(s) for {forecast_range}."
            )
            continue

    # After the loop, check final_date_output_dir
    if not success or final_date_output_dir is None:
        print(
            "No complete discharge data found after all attempts, try few hours or a day back!"
        )
        try:
            netcdf_root = os.path.join(download_dir, "netCDF")
            if os.path.exists(netcdf_root):
                shutil.rmtree(netcdf_root)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        return

    filter_csv_file_path = os.path.join(output_dir, output_csv_filename)
    CSVFILES = os.path.join(download_dir, "csvFiles")
    os.makedirs(CSVFILES, exist_ok=True)
    filter_df = pd.read_csv(filter_csv_file_path)

    # Process files from the *successful* directory
    if os.path.exists(final_date_output_dir):
        for root, _, files in os.walk(final_date_output_dir):
            for filename in files:
                if filename.endswith(".nc"):
                    netcdf_file_path = os.path.join(root, filename)
                    processnetCDF(netcdf_file_path, filter_df, CSVFILES)

    # Pass the current_download_hour and current_download_date that were successful
    ProcessForecasts(
        CSVFILES,
        current_download_date,
        current_download_hour,
        forecast_range,
        sort_by,
        data_dir,
        HUC,
    )

    print(f"The final discharge values saved to {data_dir}")
    try:
        cleanup_download_tree(download_dir, final_date_output_dir, CSVFILES)
    except Exception as e:
        print(f"Cleanup warning: {e}")


def getNWMForecasteddata(
    huc, forecast_range, forecast_date=None, hour=None, sort_by="maximum"
):
    code_dir, data_dir, output_dir = setup_directories()
    download_dir = os.path.join(
        output_dir, f"flood_{huc}", "discharge", f"{forecast_range}_forecast"
    )
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    featureIDs = Path(output_dir, f"flood_{huc}", "feature_IDs.csv")
    main(
        download_dir,
        featureIDs,
        huc,
        data_dir,
        output_dir,
        forecast_range,
        forecast_date,
        hour,
        sort_by,
    )
