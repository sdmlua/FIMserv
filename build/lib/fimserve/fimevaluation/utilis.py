"""This utility function contains how to retrieve all the necessary metadata from the s3 bucket during evaluation
for HAND FIM model outputs along with other supporting functions"""

from __future__ import annotations
import os, re, json, datetime as dt
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
    ymd = rec.get("date_ymd")
    if isinstance(ymd, str):
        try:
            return dt.date.fromisoformat(ymd)
        except Exception:
            pass
    raw = rec.get("date_of_flood")
    if isinstance(raw, str) and len(raw) >= 8:
        try:
            return dt.datetime.strptime(raw[:8], "%Y%m%d").date()
        except Exception:
            return None
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
    raw = rec.get("date_of_flood")
    if isinstance(raw, str) and "T" in raw and len(raw) >= 11:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}T{raw.split('T',1)[1][:2]}"
    ymd = rec.get("date_ymd")
    if isinstance(ymd, str) and _YMD_RE.match(ymd):
        return ymd
    if isinstance(raw, str) and len(raw) >= 8:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return "unknown"


def _context_str(
    huc8: Optional[str] = None,
    date_input: Optional[str] = None,
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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

    return ", ".join(parts) if parts else "your filters"


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
        tier = r.get("tier") or r.get("quality") or "Unknown"
        date_str = _pretty_date_for_print(r)
        res = r.get("resolution_m")
        res_txt = f"{res}m" if res is not None else "NA"
        fname = r.get("file_name") or "NA"
        blocks.append(
            f"Data Tier: {tier}\n"
            f"Benchmark FIM date: {date_str}\n"
            f"Spatial Resolution: {res_txt}\n"
            f"Raster Filename in DB: {fname}"
        )

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


# Search FIMs record in database
def find_fims(
    records: List[Dict[str, Any]],
    huc8: str,
    date_input: Optional[str] = None,
    file_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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

    Relaxed (relaxed_for_print=True) — **printing only**:
      - If start_date/end_date: ALL HUC records whose record-day is in [start, end] (hour ignored)
      - Else if date_input (day-only): ALL records that day (day-only + hourly)
      - Else (date with hour, or no date): fall back to strict behavior
    """
    huc8 = str(huc8).strip()
    recs = [r for r in records if str(r.get("huc8", "")).strip() == huc8]

    if file_name:
        fname = file_name.strip()
        recs = [r for r in recs if str(r.get("file_name", "")).strip() == fname]

    if not relaxed_for_print:
        if date_input is None:
            return recs
        target_day = _to_date(date_input)
        target_hour = _to_hour_or_none(date_input)
        out: List[Dict[str, Any]] = []
        for r in recs:
            r_day = _record_day(r)
            if r_day != target_day:
                continue
            r_hour = _record_hour_or_none(r)
            if target_hour is None:
                if r_hour is None:
                    out.append(r)
            else:
                if r_hour is not None and r_hour == target_hour:
                    out.append(r)
        return out

    # relaxed_for_print=True
    if start_date or end_date:
        d0 = _to_date(start_date) if start_date else None
        d1 = _to_date(end_date) if end_date else None
        out: List[Dict[str, Any]] = []
        for r in recs:
            r_day = _record_day(r)
            if not r_day:
                continue
            if d0 and r_day < d0:
                continue
            if d1 and r_day > d1:
                continue
            out.append(r)
        out.sort(
            key=lambda x: (str(x.get("date_of_flood", "")), str(x.get("file_name", "")))
        )
        return out

    if date_input and _to_hour_or_none(date_input) is None:
        target_day = _to_date(date_input)
        out = []
        for r in recs:
            r_day = _record_day(r)
            if r_day == target_day:
                out.append(r)
        out.sort(
            key=lambda x: (str(x.get("date_of_flood", "")), str(x.get("file_name", "")))
        )
        return out

    return find_fims(
        records=recs,
        huc8=huc8,
        date_input=date_input,
        file_name=None,
        start_date=None,
        end_date=None,
        relaxed_for_print=False,
    )


def summarize_huc_availability(records: List[Dict[str, Any]], huc8: str) -> str:
    huc8 = str(huc8).strip()
    recs = [r for r in records if str(r.get("huc8", "")).strip() == huc8]
    if not recs:
        return f"No benchmark FIMs on HUC {huc8}."

    with_raw = []
    for r in recs:
        raw = r.get("date_of_flood")
        if isinstance(raw, str) and (len(raw) == 8 or ("T" in raw and len(raw) >= 11)):
            with_raw.append(r)

    if not with_raw:
        rps = sorted(
            {str(r.get("return_period")) for r in recs if r.get("return_period")}
        )
        if rps:
            return f"No real flood-based benchmarks on HUC {huc8}. Only synthetic return periods available: {', '.join(rps)}."
        return f"No real flood-based benchmarks on HUC {huc8}."

    day_set, hour_set = set(), set()
    for r in with_raw:
        d = _record_day(r)
        if d:
            day_set.add(d.isoformat())
            h = _record_hour_or_none(r)
            if h is not None:
                hour_set.add(f"{d:%Y-%m-%d}T{h:02d}")

    parts = []
    if day_set:
        parts.append("days: " + ", ".join(sorted(day_set)))
    if hour_set:
        parts.append("hourly: " + ", ".join(sorted(hour_set)))
    return f"Available benchmark dates on HUC {huc8}: " + " | ".join(parts)


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


def download_fim_assets(record: Dict[str, Any], dest_dir: str) -> Dict[str, Any]:
    """
    Download the .tif (if present) and any .gpkg from the record's folder to dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    out = {"tif": None, "gpkg_files": []}

    # TIF
    tif_key = _tif_key_from_record(record)
    if tif_key:
        local = os.path.join(dest_dir, os.path.basename(tif_key))
        if not os.path.exists(local):
            _download(BUCKET, tif_key, local)
        out["tif"] = local

    # GPKGs (list folder)
    folder = _folder_from_record(record)
    for key in _list_prefix(folder):
        if key.lower().endswith(".gpkg"):
            local = os.path.join(dest_dir, os.path.basename(key))
            if not os.path.exists(local):
                _download(BUCKET, key, local)
            out["gpkg_files"].append(local)

    return out


# Make the huc_event_dict for FIM generation for multiple events on one HUC
def build_huc_event_dict(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for r in records:
        huc = str(r.get("huc8"))
        day = _record_day(r)
        if not day:
            continue
        hour = _record_hour_or_none(r)
        ts = day.isoformat() if hour is None else f"{day:%Y-%m-%d} {hour:02d}:00:00"
        d.setdefault(huc, []).append(ts)
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

    # STRICT set (used for status/logic/downloads)
    strict_matches = find_fims(
        records,
        huc8=HUC8,
        date_input=date_input,
        file_name=file_name,
        start_date=None,
        end_date=None,
        relaxed_for_print=False,
    )

    # RELAXED set (printing only)
    relaxed_records_for_print = find_fims(
        records,
        huc8=HUC8,
        date_input=date_input,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
        relaxed_for_print=True,
    )

    ctx = _context_str(
        huc8=HUC8,
        date_input=date_input,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
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
            )
            else "not_found"
        )
        return {
            "status": status,
            "message": (
                f"No match for HUC {HUC8}"
                + (f" and '{date_input}'" if date_input else "")
                + (f" and file '{file_name}'" if file_name else "")
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
        dls.append({"record": r, "downloads": download_fim_assets(r, out_dir)})

    return {
        "status": "ok",
        "message": f"Downloaded {len(dls)} record(s) to '{out_dir}'.",
        "matches": dls,
        "printable": printable,
    }
