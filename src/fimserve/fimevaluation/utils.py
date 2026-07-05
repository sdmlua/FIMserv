"""
Author: Supath Dhital (sdhital@ua.edu)
Updated: March 2026

This utility function contains how to retrieve all the necessary metadata from the s3 bucket during evaluation
for HAND FIM model outputs along with other supporting functions

"""

from __future__ import annotations
import os
import re
import json
import datetime as dt
from typing import List, Dict, Any, Optional

import urllib.parse
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# constants
BUCKET = "sdmlab"
CATALOG_KEY = (
    "FIM_Database/FIM_Viz/catalog_core.json"  # Path of the json file in the s3 bucket
)

# s3 client
_S3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


# helpers for direct S3 file links
def s3_http_url(bucket: str, key: str) -> str:
    """Build a public-style S3 HTTPS URL."""
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key, safe='/')}"


# utils
_YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YMD_COMPACT_RE = re.compile(r"^\d{8}$")
_YMDH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}$")
_YMDHMS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$")


def _normalize_user_dt(s: str) -> str:
    s = s.strip()
    s = s.replace("/", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _to_date(s: str) -> dt.date:
    s = _normalize_user_dt(s)
    if _YMD_COMPACT_RE.match(s):
        return dt.datetime.strptime(s, "%Y%m%d").date()
    if _YMD_RE.match(s):
        return dt.date.fromisoformat(s)
    try:
        return dt.datetime.fromisoformat(s).date()
    except Exception:
        m = re.match(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2})$", s)
        if m:
            return dt.datetime.fromisoformat(f"{m.group(1)} {m.group(2)}:00:00").date()
        raise ValueError(f"Bad date format: {s}")


def _to_hour_or_none(s: str) -> Optional[int]:
    s = _normalize_user_dt(s)
    if _YMD_RE.match(s) or _YMD_COMPACT_RE.match(s):
        return None
    m = re.match(r"^\d{4}-\d{2}-\d{2}[ T](\d{2})$", s)
    if m:
        return int(m.group(1))
    try:
        dt_obj = dt.datetime.fromisoformat(s)
        return dt_obj.hour
    except Exception:
        m2 = re.match(r"^\d{4}-\d{2}-\d{2}T(\d{2})$", s)
        if m2:
            return int(m2.group(1))
        return None


def _record_day(rec: Dict[str, Any]) -> Optional[dt.date]:
    # standard date_ymd, event ts, start and end date for high water marks
    if rec.get("date_ymd"):
        try:
            return dt.date.fromisoformat(rec["date_ymd"])
        except Exception:
            pass
    if rec.get("event_ts") and len(str(rec["event_ts"])) >= 8:
        try:
            return dt.datetime.strptime(str(rec["event_ts"])[:8], "%Y%m%d").date()
        except Exception:
            pass
    if rec.get("start_date_ymd"):
        try:
            return dt.date.fromisoformat(rec["start_date_ymd"])
        except Exception:
            pass
    # Fallback
    raw = rec.get("date_of_flood")
    if isinstance(raw, str) and len(raw) >= 8:
        try:
            return dt.datetime.strptime(raw[:8], "%Y%m%d").date()
        except Exception:
            pass

    return None


def _record_hour_or_none(rec: Dict[str, Any]) -> Optional[int]:
    raw = rec.get("date_of_flood")
    if isinstance(raw, str) and "T" in raw and len(raw) >= 11:
        try:
            return int(raw.split("T", 1)[1][:2])
        except Exception:
            return None
    return None


# Printing helpers
def _pretty_date_for_print(rec: Dict[str, Any]) -> str:
    start = rec.get("start_date_ymd")
    end = rec.get("end_date_ymd")
    if start and end:
        return f"{start} to {end}"
    raw = rec.get("date_of_flood")
    if isinstance(raw, str) and "T" in raw and len(raw) >= 11:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}T{raw.split('T', 1)[1][:2]}"
    ymd = rec.get("date_ymd")
    if isinstance(ymd, str) and _YMD_RE.match(ymd):
        return ymd
    ts = rec.get("event_ts")
    if ts and len(str(ts)) >= 8:
        s_ts = str(ts)
        return f"{s_ts[:4]}-{s_ts[4:6]}-{s_ts[6:8]}"
    if isinstance(raw, str) and len(raw) >= 8:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return "unknown"


def _context_str(
    huc8: Optional[str] = None,
    date_input: Optional[str] = None,
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_period: Optional[int] = None,
) -> str:
    """
    Builds a readable context summary for printing headers.
    Example outputs:
      - "HUC 12090301"
      - "HUC 12090301, date '2017-08-30'"
      - "HUC 12090301, range 2017-08-30 to 2017-09-01"
      - "HUC 12090301, file 'PSS_3_0m_20170830T162251_BM.tif'"
    """
    parts = []
    if huc8:
        parts.append(f"HUC {huc8}")
    if date_input:
        parts.append(f"date '{date_input}'")
    if start_date or end_date:
        if start_date and end_date:
            parts.append(f"range {start_date} to {end_date}")
        elif start_date:
            parts.append(f"from {start_date}")
        elif end_date:
            parts.append(f"until {end_date}")
    if file_name:
        parts.append(f"file '{file_name}'")

    if return_period is not None:
        parts.append(f"return period {int(return_period)}")

    return ", ".join(parts) if parts else "your filters"


def _normalize_tier_for_comparison(s: str) -> str:
    """Helper to make tier filtering case/format insensitive."""
    if not s:
        return ""
    return str(s).lower().replace(" ", "").replace("-", "").replace("_", "")


def format_records_for_print(
    records: List[Dict[str, Any]], context: Optional[str] = None
) -> str:
    if not records:
        ctx = context or "your filters"
        return f"Benchmark FIMs were not matched for {ctx}."

    header = (
        f"Following are the available benchmark data for {context}:\n"
        if context
        else ""
    )
    blocks: List[str] = []

    for r in records:
        tier = _tier_label(r)
        date_str = _pretty_date_for_print(r)
        res = r.get("resolution_m")
        res_txt = f"{res}m" if res is not None else "NA"
        fname = r.get("file_name") or "NA"
        huc_list = _record_huc8_list(r)
        area_map = r.get("huc_area_results", {})

        huc_display = []
        for h in huc_list:
            if h in area_map:
                huc_display.append(f"{h} (Intersected area: {area_map[h]:.2f}%)")
            else:
                huc_display.append(h)

        huc_str = ", ".join(huc_display) if huc_display else "NA"

        lines = [
            f"Data Tier: {tier}",
            f"Spatial Resolution: {res_txt}",
            f"Intersecting HUC8: {huc_str}",
            f"Raster Filename in DB: {fname}",
        ]

        if date_str != "unknown":
            lines.insert(1, f"Benchmark FIM date: {date_str}")
        elif r.get("return_period") is not None:
            lines.insert(1, f"Return Period: {r.get('return_period')}yr")
        else:
            lines.insert(1, "Benchmark FIM date: unknown")

        blocks.append("\n".join(lines))

    return (header + "\n\n".join(blocks)).strip()


# S3 and json catalog
def load_catalog_core() -> Dict[str, Any]:
    obj = _S3.get_object(Bucket=BUCKET, Key=CATALOG_KEY)
    return json.loads(obj["Body"].read().decode("utf-8", "replace"))


def _list_prefix(prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = _S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    return keys


def _download(bucket: str, key: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    _S3.download_file(bucket, key, dest_path)
    return dest_path


def _record_huc8_list(rec: Dict[str, Any]) -> List[str]:
    """
    Return HUC8s from a record as a normalized list of strings.

    Catalog store:
      - "huc8": "['03020201','03020202',...]"
    """
    v = rec.get("huc8")

    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        out: List[str] = []
        for x in v:
            if x is None:
                continue
            s = str(x).strip().strip("'").strip('"')
            if s:
                out.append(s)
        return out

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []

        # stringified list like "['03020201', '03020202']"
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    out: List[str] = []
                    for x in parsed:
                        if x is None:
                            continue
                        t = str(x).strip().strip("'").strip('"')
                        if t:
                            out.append(t)
                    return out
            except Exception:
                pass

            inner = s[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            out2: List[str] = []
            for p in parts:
                t = p.strip().strip("'").strip('"')
                if t:
                    out2.append(t)
            return out2
        return [s.strip().strip("'").strip('"')]
    return [str(v).strip()]


def _tier_label(rec: Dict[str, Any]) -> str:
    """
    Normalize tier/quality/HWM style labels into a consistent printable string.
    """
    raw = rec.get("tier")
    if raw is None or str(raw).strip() == "":
        raw = rec.get("quality")
    if raw is None or str(raw).strip() == "":
        raw = rec.get("HWM")  # For High Water Marks (HWM)

    s = str(raw).strip() if raw is not None else ""
    if not s:
        return "Unknown"

    s_low = s.lower().replace(" ", "").replace("-", "_")
    # forms: Tier_2, tier2, 2, Tier 2
    if "tier" in s_low:
        m = re.search(r"tier[_ ]*(\d+)", s_low)
        if m:
            return f"Tier {m.group(1)}"
        return s.replace("_", " ").strip()
    if s.isdigit():
        return f"Tier {s}"
    return s


# finding the benchmark FIMs
def find_fims(
    records: List[Dict[str, Any]],
    huc8: str,
    date_input: Optional[str] = None,
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_period: Optional[int] = None,
    tier: Optional[str] = None,
    relaxed_for_print: bool = False,
) -> List[Dict[str, Any]]:
    """
    Unified finder.

    Strict (relaxed_for_print=False) — use for downloads/process:
      - Filter by HUC8
      - If file_name: exact filename
      - If date_input:
          * date-only -> ONLY day-only benchmarks
          * date+hour -> ONLY exact-hour benchmarks
      - Ignores start_date/end_date
      - return_period - is especially for the BLE HUC 8 level dataset, which correponsing flows are saved within the s3 bucket

    Relaxed (relaxed_for_print=True) — **printing only**:
      - If start_date/end_date: ALL HUC records whose record-day is in [start, end] (hour ignored)
      - Else if date_input (day-only): ALL records that day (day-only + hourly)
      - Else (date with hour, or no date): fall back to strict behavior
    """
    huc8 = str(huc8).strip()
    recs = [r for r in records if huc8 in set(_record_huc8_list(r))]

    # Applying tier if passed
    if tier:
        target_t = _normalize_tier_for_comparison(tier)
        recs = [
            r
            for r in recs
            if _normalize_tier_for_comparison(_tier_label(r)) == target_t
        ]

    # Filter by Return Period (Synthetic/Tier 4)
    if return_period is not None:
        trp = int(return_period)
        recs = [
            r for r in recs if r.get("return_period") and int(r["return_period"]) == trp
        ]

    # Filter by Filename
    if file_name:
        fname = file_name.strip()
        recs = [r for r in recs if str(r.get("file_name", "")).strip() == fname]

    # STRICT SEARCH
    # Used for processing and specific downloads.
    if not relaxed_for_print:
        if date_input is None:
            return recs

        target_day = _to_date(date_input)
        target_hour = _to_hour_or_none(date_input)
        out = []

        for r in recs:
            # Check for HWM Range Overlap first
            r_start = r.get("start_date_ymd")
            r_end = r.get("end_date_ymd")

            if r_start and r_end:
                # If target_day falls within the HWM range, it's a match
                if (
                    dt.date.fromisoformat(r_start)
                    <= target_day
                    <= dt.date.fromisoformat(r_end)
                ):
                    out.append(r)
                continue

            # Check for Exact Real-Event Match
            r_day = _record_day(r)
            if r_day == target_day:
                # If searching by hour, verify hour match too
                if target_hour is not None:
                    r_hour = _record_hour_or_none(r)
                    if r_hour == target_hour:
                        out.append(r)
                else:
                    # Searching by day only, and record is day-only
                    if _record_hour_or_none(r) is None:
                        out.append(r)
        return out

    # RELAXED SEARCH
    # Gives all the available
    else:
        # Determine the search window
        d0 = _to_date(start_date) if start_date else None
        d1 = _to_date(end_date) if end_date else None

        if not d0 and not d1 and date_input:
            d0 = d1 = _to_date(date_input)

        if not d0 and not d1:
            return recs

        out = []
        for r in recs:
            r_day = _record_day(r)
            r_start_str = r.get("start_date_ymd")
            r_end_str = r.get("end_date_ymd")

            # Check Intersection with HWM Range
            if r_start_str and r_end_str:
                rs = dt.date.fromisoformat(r_start_str)
                re = dt.date.fromisoformat(r_end_str)
                if (not d1 or rs <= d1) and (not d0 or re >= d0):
                    out.append(r)

            elif r_day:
                if (not d0 or r_day >= d0) and (not d1 or r_day <= d1):
                    out.append(r)

        out.sort(key=lambda x: str(_record_day(x) or ""))
        return out


def summarize_huc_availability(records: List[Dict[str, Any]], huc8: str) -> str:
    huc8 = str(huc8).strip()
    recs = [r for r in records if huc8 in set(_record_huc8_list(r))]
    if not recs:
        return f"No benchmark FIMs on HUC {huc8}."

    # Find all records that have some form of real-world date/range
    real_benchmarks = []
    for r in recs:
        if (
            r.get("date_of_flood")
            or r.get("date_ymd")
            or r.get("start_date_ymd")
            or r.get("event_ts")
        ):
            real_benchmarks.append(r)

    if not real_benchmarks:
        rps = sorted(
            {str(r.get("return_period")) for r in recs if r.get("return_period")}
        )
        if rps:
            return (
                f"No real flood-based benchmarks on HUC {huc8}. "
                f"Only synthetic return periods available: {', '.join(rps)}."
            )
        return f"No real flood-based benchmarks on HUC {huc8}."

    # Collect unique dates/ranges to show the user
    avail_info = set()
    for r in real_benchmarks:
        date_label = _pretty_date_for_print(r)
        if date_label != "unknown":
            avail_info.add(date_label)

    return f"Available benchmark dates/ranges on HUC {huc8}: " + " | ".join(
        sorted(avail_info)
    )


# Get the files from s3 bucket
def _folder_from_record(rec: Dict[str, Any]) -> str:
    s3_key = rec.get("s3_key")
    if not s3_key or "/" not in s3_key:
        raise ValueError("Record lacks s3_key to derive folder")
    return s3_key.rsplit("/", 1)[0] + "/"


def _tif_key_from_record(rec: Dict[str, Any]) -> Optional[str]:
    tif_url = rec.get("tif_url")
    if isinstance(tif_url, str) and ".amazonaws.com/" in tif_url:
        return tif_url.split(".amazonaws.com/", 1)[1]
    fname = rec.get("file_name")
    if not fname:
        return None
    return _folder_from_record(rec) + fname


def download_fim_assets(
    record, dest_dir, return_period=None, download_flows: bool = True
) -> Dict[str, Any]:
    """
    Download the .tif (if present) and any .gpkg from the record's folder to dest_dir.
    Optionally download FLOWS CSVs (Tier_4) if download_flows=True.
    """
    os.makedirs(dest_dir, exist_ok=True)
    out = {"tif": None, "gpkg_files": [], "flow_csv_files": []}

    # TIF
    tif_key = _tif_key_from_record(record)
    if tif_key:
        local = os.path.join(dest_dir, os.path.basename(tif_key))
        if not os.path.exists(local):
            _download(BUCKET, tif_key, local)
        out["tif"] = local

    # GPKGs
    folder = _folder_from_record(record)
    keys = _list_prefix(folder)

    for key in keys:
        if key.lower().endswith(".gpkg"):
            local = os.path.join(dest_dir, os.path.basename(key))
            if not os.path.exists(local):
                _download(BUCKET, key, local)
            out["gpkg_files"].append(local)

    # FLOWS CSVs-->optional
    if download_flows:
        target_tag = (
            None if return_period is None else f"FLOWS_{int(return_period)}YR".upper()
        )
        for key in keys:
            if not key.lower().endswith(".csv"):
                continue
            base = os.path.basename(key)

            # Based on user asked Return period, only pull that one
            if target_tag and target_tag not in base.upper():
                continue
            if (return_period is None) and ("FLOWS_" not in base.upper()):
                continue

            local = os.path.join(dest_dir, base)
            if not os.path.exists(local):
                _download(BUCKET, key, local)
            out["flow_csv_files"].append(local)

    return out


# Make the huc_event_dict for FIM generation for multiple events on one HUC
def build_huc_event_dict(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for r in records:
        hucs = _record_huc8_list(r)
        if not hucs:
            continue
        day = _record_day(r)
        if not day:
            continue
        hour = _record_hour_or_none(r)
        ts = day.isoformat() if hour is None else f"{day:%Y-%m-%d} {hour:02d}:00:00"
        for huc in hucs:
            d.setdefault(str(huc), []).append(ts)
    for k in list(d.keys()):
        d[k] = sorted(set(d[k]))
    return d


def availability(HUC8: str) -> str:
    catalog = load_catalog_core()
    return summarize_huc_availability(catalog.get("records", []), HUC8)


# benchmark FIM find and download function
def bmFIMFindandDownload(
    HUC8: str,
    out_dir: Optional[str] = None,
    date_input: Optional[str] = None,
    download: bool = False,
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_period: Optional[int] = None,
    tier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - Loads catalog from S3
    - Strictly finds matches by HUC (+ optional day/hour) and optional file_name for logic/downloads
    - Builds a relaxed list for printing (date-only => day+hour; or date range)
    - If download=False: DO NOT download. Return matches + availability text.
    - If download=True: require out_dir; download .tif and any .gpkg for STRICT matches only.
    """
    catalog = load_catalog_core()
    records = catalog.get("records", [])

    # STRICT set
    strict_matches = find_fims(
        records,
        huc8=HUC8,
        date_input=date_input,
        file_name=file_name,
        start_date=None,
        end_date=None,
        relaxed_for_print=False,
        return_period=return_period,
        tier=tier,
    )

    # RELAXED set
    relaxed_records_for_print = find_fims(
        records,
        huc8=HUC8,
        date_input=date_input,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
        relaxed_for_print=True,
        return_period=return_period,
        tier=tier,
    )

    ctx = _context_str(
        huc8=HUC8,
        date_input=date_input,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
        return_period=return_period,
    )
    printable = format_records_for_print(relaxed_records_for_print, context=ctx)

    # No strict matches
    if not strict_matches:
        status = (
            "info"
            if (
                date_input is None
                and file_name is None
                and not start_date
                and not end_date
                and not return_period
            )
            else "not_found"
        )
        return {
            "status": status,
            "message": (
                f"No match for HUC {HUC8}"
                + (f" and '{date_input}'" if date_input else "")
                + (f" and file '{file_name}'" if file_name else "")
                + (f" and return period '{return_period}'" if return_period else "")
                + ".\n"
                + summarize_huc_availability(records, HUC8)
            ),
            "matches": [],
            "printable": printable,
        }

    # If not downloading, return pretty print + strict metadata
    if not download:
        return {
            "status": "ok",
            "message": (
                f"Found {len(strict_matches)} record(s) for HUC {HUC8}"
                + (f" and '{date_input}'" if date_input else "")
                + (f" and file '{file_name}'" if file_name else "")
                + (f" and return period '{return_period}'" if return_period else "")
                + ".\n"
                + summarize_huc_availability(records, HUC8)
            ),
            "matches": [{"record": r, "downloads": None} for r in strict_matches],
            "printable": printable,
        }

    # Download mode: need out_dir
    if not out_dir:
        return {
            "status": "error",
            "message": "When download=True, you must provide out_dir.",
            "matches": [],
            "printable": printable,
        }

    dls = []
    for r in strict_matches:  # only strict matches are downloaded
        dl = download_fim_assets(r, out_dir, return_period=return_period)
        dls.append({"record": r, "downloads": dl})

    return {
        "status": "ok",
        "message": f"Downloaded {len(dls)} record(s) to '{out_dir}' with return period '{return_period}'.",
        "matches": dls,
        "printable": printable,
    }
