"""
Date Updated: 31 Mar, 2026
Author: Supath Dhital (sdhital@ua.edu)
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os
import tempfile
import shutil

# Internal utilities
from .utils import (
    load_catalog_core,
    download_fim_assets,
    _to_date,
    _to_hour_or_none,
    _record_day,
    _record_hour_or_none,
    format_records_for_print,
    find_fims,
    _record_huc8_list,
    _folder_from_record,
    _list_prefix,
    _download,
    BUCKET,
)

from ..datadownload import DownloadHUC8, setup_directories
from ..streamflowdata.nwmretrospective import getNWMretrospectivedata
from ..intersectedHUC import HUC8RESTFinder
from ..runFIM import runOWPHANDFIM


class FIMService:
    """
    - query(huc8, date_input=None, file_name=None) -> strict matches + pretty listing
    - process(..., ensure_owp, generate_owp_if_missing, base_dir=None, file_name=None)
      Creates folders {CWD}/FIM_evaluation/FIM_inputs/HUC{huc}_flood{YYYYMMDD[HHMMSS]}
      Downloads ONLY the matched record(s) (and their gpkg) into that folder.
    """

    # Run setup_directories() only when actually needed.
    def _ensure_roots(self):
        if hasattr(self, "_roots_initialized"):
            return

        _, _, out_root = setup_directories()
        self.default_root = out_root
        self.owp_root = Path(os.getenv("OWP_OUT_ROOT", out_root))

        self._roots_initialized = True

    def availability(self, HUCID: str) -> str:
        from .utils import availability as _avail

        return _avail(HUCID)

    @staticmethod
    def _site_of(rec: Dict[str, Any]) -> str:
        s = rec.get("site_id") or rec.get("site")
        s = str(s or "").strip()
        return s if s else "site_unknown"

    def _find_any_owp_for_day(self, huc8: str, ymd: str) -> Optional[Path]:
        dirp = self.owp_root / f"flood_{huc8}" / f"{huc8}_inundation"
        if not dirp.exists():
            return None
        for p in sorted(dirp.glob(f"NWM_{ymd}*_{huc8}_inundation.tif")):
            return p
        return None

    # Return-period helpers
    def _flows_inputs_dir(self) -> Path:
        """
        Fixed location to store return-period flow CSVs.
        User requested: ./data/inputs
        """
        p = Path(os.getcwd()) / "data" / "inputs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _download_return_period_flows_csv(self, huc8: str, return_period: int) -> Path:
        """
        Download the Tier_4 BLE flow CSV for this HUC8 + return period into ./data/inputs.
        This is used instead of event-based NWM retrospective download.
        """
        catalog = load_catalog_core()
        records = catalog.get("records", [])

        # Find matching records by HUC and return period (Tier_4 BLE records have no date)
        matches = find_fims(
            records=records,
            huc8=str(huc8).strip(),
            date_input=None,
            file_name=None,
            start_date=None,
            end_date=None,
            return_period=int(return_period),
            relaxed_for_print=False,
        )

        if not matches:
            raise FileNotFoundError(
                f"No benchmark record found for HUC {huc8} with return period {return_period}."
            )

        # Prefer Tier_4 records if present
        tier4 = []
        for r in matches:
            t = (
                str(r.get("tier") or r.get("quality") or "")
                .strip()
                .lower()
                .replace(" ", "")
            )
            if t.startswith("tier_4") or t.startswith("tier4"):
                tier4.append(r)
        rec = tier4[0] if tier4 else matches[0]

        # List keys in that S3 prefix/folder
        prefix = _folder_from_record(rec)
        keys = _list_prefix(prefix)

        tag = f"FLOWS_{int(return_period)}YR".upper()
        flow_keys = []
        for k in keys:
            if k.lower().endswith(".csv") and tag in os.path.basename(k).upper():
                flow_keys.append(k)

        if not flow_keys:
            raise FileNotFoundError(
                f"No {tag} CSV found under S3 prefix '{prefix}' for HUC {huc8}."
            )

        flows_dir = self._flows_inputs_dir()
        src_key = flow_keys[0]
        local = flows_dir / os.path.basename(src_key)

        if not local.exists():
            _download(BUCKET, src_key, str(local))

        return local

    # For the existing filename search
    def _find_any_owp_for_return_period(
        self, huc8: str, return_period: int
    ) -> Optional[Path]:
        """
        Look inside the inundation folder for a BLE return-period inundation tif.
        Match any tif whose name contains both the HUC8 and {return_period}YR tokens.
        e.g. BLE_HUC_11110205_FLOWS_100YR_921950W351837N_inundation.tif
        """
        dirp = self.owp_root / f"flood_{huc8}" / f"{huc8}_inundation"
        if not dirp.exists():
            return None
        rp_tag = f"{return_period}YR"
        cand = [
            p
            for p in dirp.glob("BLE*_inundation.tif")
            if huc8 in p.name and rp_tag in p.name.upper()
        ]
        if not cand:
            return None
        return sorted(cand, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    def _generate_owp_return_period(
        self,
        huc8: str,
        return_period: int,
        dest_dirs: List[str],
    ) -> Optional[str]:
        self._ensure_roots()

        rp = int(return_period)

        # If already generated, just copy to dest dirs -- same logic as the date-based path
        existing = self._find_any_owp_for_return_period(huc8, rp)
        if existing and existing.exists():
            copied_any: Optional[str] = None
            for d in dest_dirs:
                copied_any = self._copy_to_dest(existing, d)
            return copied_any

        print(f"Generating return-period HAND FIM for HUC {huc8} (RP={rp})...")
        DownloadHUC8(huc8, version="4.8")

        flows_csv = self._download_return_period_flows_csv(huc8, rp)
        print(f"Downloaded return-period flows CSV to '{flows_csv}'.")

        runOWPHANDFIM(huc8)

        # After run, re-check the inundation folder for the produced tif
        produced = self._find_any_owp_for_return_period(huc8, rp)
        if not produced:
            return None

        # Copy produced tif into each destination folder
        copied_any: Optional[str] = None
        for d in dest_dirs:
            copied_any = self._copy_to_dest(produced, d)

        return copied_any

    # HWM Range helpers
    def _expected_owp_path_hwm(self, huc8: str, start: str, end: str) -> Path:
        """
        Naming convention for HWM range-based maximum discharge maps.
        Format: NWM_STARTDATE_ENDDATE_maximum_HUC_inundation.tif
        """
        s, e = start.replace("-", ""), end.replace("-", "")
        name = f"NWM_{s}_{e}_maximum_{huc8}_inundation.tif"
        return self.owp_root / f"flood_{huc8}" / f"{huc8}_inundation" / name

    # Query
    def query(
        self,
        HUCID: str,
        date_input: Optional[str] = None,
        file_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        return_period: Optional[int] = None,
        tier: Optional[str] = None,
        huc_intersectedarea: bool = False,
        huc_thresholdarea: float = 0.0,
    ) -> Dict[str, Any]:
        catalog = load_catalog_core()
        records = catalog.get("records", [])
        huc8 = str(HUCID).strip()

        # strict set
        strict_matches = find_fims(
            records,
            huc8=huc8,
            date_input=date_input,
            file_name=file_name,
            start_date=start_date,
            end_date=end_date,
            return_period=return_period,
            tier=tier,
            relaxed_for_print=False,
        )

        # relaxed set
        relaxed_matches = find_fims(
            records,
            huc8=huc8,
            date_input=date_input,
            file_name=file_name,
            start_date=start_date,
            end_date=end_date,
            return_period=return_period,
            tier=tier,
            relaxed_for_print=True,
        )

        if huc_intersectedarea and relaxed_matches:
            finder = HUC8RESTFinder(debug=False)
            for rec in relaxed_matches:
                gpkg_url = rec.get("gpkg_url")
                if not gpkg_url:
                    continue

                # Derive S3 Key from URL
                s3_key = gpkg_url.split(".amazonaws.com/")[1]

                with tempfile.NamedTemporaryFile(suffix=".gpkg") as tmp:
                    # Download only the small AOI polygon
                    _download(BUCKET, s3_key, tmp.name)
                    # Run REST-based mapping and attach to record
                    rec["huc_area_results"] = finder.get_huc_area_mapping(tmp.name)

        status = (
            "ok"
            if strict_matches
            else (
                "info"
                if (
                    date_input is None
                    and file_name is None
                    and not start_date
                    and not end_date
                )
                else "not_found"
            )
        )

        base_msg = f"Found {len(strict_matches)} record(s) for HUC {huc8}"
        if date_input:
            base_msg += f" and '{date_input}'"
        if file_name:
            base_msg += f" and file '{file_name}'"
        if start_date or end_date:
            base_msg += f" in range [{start_date or '-∞'} , {end_date or '∞'}]"

        printable = format_records_for_print(relaxed_matches)
        return {
            "status": status,
            "message": base_msg + "\n" + self.availability(huc8),
            "matches": strict_matches,
            "printable": printable,
        }

    def process(
        self,
        huc8: Optional[str] = None,
        date_input: Optional[str] = None,
        ensure_owp: bool = True,
        generate_owp_if_missing: bool = True,
        out_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        return_period: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tier: Optional[str] = None,
        huc_thresholdarea: float = 0.0,
        eval_individual_huc: bool = False,
    ) -> Dict[str, Any]:
        # Initialize roots and load benchmark catalog data
        self._ensure_roots()
        catalog = load_catalog_core()
        records = catalog.get("records", [])

        target_recs = []
        huc8_list = []

        # Resolve target record and associated HUC list based on filename or HUC ID
        if file_name:
            fname = file_name.strip()
            found = [r for r in records if str(r.get("file_name", "")).strip() == fname]

            # Apply tier filtering to narrow down the specific benchmark file
            if tier:
                from .utils import _normalize_tier_for_comparison

                target_t = _normalize_tier_for_comparison(tier)
                found = [
                    r
                    for r in found
                    if _normalize_tier_for_comparison(r.get("tier") or r.get("quality"))
                    == target_t
                ]

            if not found:
                return {
                    "status": "not_found",
                    "message": f"File '{file_name}' not found.",
                    "folders": [],
                }

            target_rec = found[0]
            target_recs = [target_rec]

            # Filter HUCs using ArcGIS REST API if area threshold is provided
            if huc_thresholdarea > 0:
                finder = HUC8RESTFinder(debug=False)
                gpkg_url = target_rec.get("gpkg_url")
                if gpkg_url:
                    s3_key = gpkg_url.split(".amazonaws.com/")[1]
                    with tempfile.NamedTemporaryFile(suffix=".gpkg") as tmp:
                        _download(BUCKET, s3_key, tmp.name)
                        area_map = finder.get_huc_area_mapping(tmp.name)
                        huc8_list = [
                            h for h, pct in area_map.items() if pct >= huc_thresholdarea
                        ]
                else:
                    huc8_list = _record_huc8_list(target_rec)
            else:
                # Fallback to all HUCs defined in the benchmark JSON
                huc8_list = (
                    [huc8]
                    if (huc8 and huc8.strip() != "")
                    else _record_huc8_list(target_rec)
                )
        else:
            if not huc8:
                return {
                    "status": "error",
                    "message": "Provide HUCID or file_name.",
                    "folders": [],
                }
            huc8_list = [str(huc8).strip()]

        # Infer temporal metadata for NWM retrospective if not explicitly provided
        if target_recs and not any([date_input, start_date, end_date, return_period]):
            rec = target_recs[0]
            r_tier = str(rec.get("tier") or rec.get("quality") or "").upper()
            if "HWM" in r_tier:
                start_date, end_date = (
                    rec.get("start_date_ymd"),
                    rec.get("end_date_ymd"),
                )
            elif "TIER 4" in r_tier or "TIER4" in r_tier:
                return_period = rec.get("return_period")
            else:
                date_input = rec.get("date_ymd") or str(rec.get("event_ts", ""))[:8]

        # Set up output root and identify benchmark site
        inputs_root = Path(out_dir) if out_dir else Path.cwd() / "FIMevaluation_inputs"
        inputs_root.mkdir(parents=True, exist_ok=True)
        site = self._site_of(target_recs[0]) if target_recs else "site_unknown"

        folders_out = []
        generated_tif_paths = []

        # Setup master folder for combined evaluation mode
        if not eval_individual_huc:
            target_folder = inputs_root / f"All_HUC_MOSAICED_{site}"
            target_folder.mkdir(parents=True, exist_ok=True)
            if target_recs:
                download_fim_assets(
                    target_recs[0], str(target_folder), return_period=return_period
                )

        # Loop through HUCs to generate HAND FIM inundation maps
        for h_id in huc8_list:
            if eval_individual_huc:
                target_folder = inputs_root / f"HUC{h_id}_{site}"
                target_folder.mkdir(parents=True, exist_ok=True)
                if target_recs:
                    download_fim_assets(
                        target_recs[0], str(target_folder), return_period=return_period
                    )

            print(f"Generating HAND FIM for HUC {h_id}...")
            owp_path = (
                self._ensure_owp_to(
                    huc8=h_id,
                    user_dt=date_input,
                    dest_dir=str(target_folder),
                    generate_if_missing=generate_owp_if_missing,
                    return_period=return_period,
                    start_date=start_date,
                    end_date=end_date,
                )
                if ensure_owp
                else None
            )

            if owp_path:
                generated_tif_paths.append(Path(owp_path))
                folders_out.append(
                    {"huc": h_id, "folder": str(target_folder), "owp_path": owp_path}
                )

        # Mosaic all generated HUC rasters into a single descriptive file
        if not eval_individual_huc and len(generated_tif_paths) > 1:
            import rasterio
            from rasterio.merge import merge

            print(
                f"Mosaicking {len(generated_tif_paths)} rasters into descriptive composite..."
            )
            src_files = [rasterio.open(p) for p in generated_tif_paths]
            mosaic, out_trans = merge(src_files)

            # Configure final metadata with LZW compression and tiling
            out_meta = src_files[0].meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                }
            )

            # Create the final descriptive filename
            original_name = generated_tif_paths[0].name
            mosaic_name = original_name.replace(huc8_list[0], "mosaicked_allhuc")
            mosaic_path = target_folder / mosaic_name

            with rasterio.open(mosaic_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            for src in src_files:
                src.close()

            # Clean up individual HUC rasters to keep the folder tidy
            print("Cleaning up intermediate individual HUC rasters...")
            for p in generated_tif_paths:
                if p.exists():
                    os.remove(p)

            print(f"Final mosaicked raster saved to: {mosaic_path.name}")
            msg = f"Processed {len(huc8_list)} HUC(s) into one mosaicked file: {mosaic_path.name}"
        else:
            msg = f"Processed {len(huc8_list)} HUC(s)."
        return {"status": "ok", "message": msg, "folders": folders_out}

    # Internals
    @staticmethod
    def _date_label_for_record(rec: Dict[str, Any]) -> str:
        day = _record_day(rec)
        hh = _record_hour_or_none(rec)
        if not day:
            raw = str(rec.get("date_raw", ""))[:8]
            return raw if raw else "unknown"
        if hh is None:
            return f"{day:%Y%m%d}"
        return f"{day:%Y%m%d}{hh:02d}0000"

    @staticmethod
    def _date_label_from_user(user_dt: str) -> str:
        day = _to_date(user_dt)
        hh = _to_hour_or_none(user_dt)
        return f"{day:%Y%m%d}" if hh is None else f"{day:%Y%m%d}{hh:02d}0000"

    def _user_dt_from_label(self, label: str) -> Optional[str]:
        """Convert a YYYYMMDD string back into a dash-separated user string."""
        try:
            if len(label) == 8:
                return f"{label[:4]}-{label[4:6]}-{label[6:]}"
            elif len(label) >= 10:
                return f"{label[:4]}-{label[4:6]}-{label[6:8]}T{label[8:10]}"
        except Exception:
            pass
        return None

    def _ensure_owp_to(
        self,
        huc8: str,
        user_dt: Optional[str],
        dest_dir: str,
        generate_if_missing: bool,
        return_period: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[str]:
        """
        Idempotent ensure:
        - Checks for HWM Range maps first.
        - Checks for specific timestamps or daily maps.
        """
        # HWM Range Map
        if start_date and end_date:
            existing = self._expected_owp_path_hwm(huc8, start_date, end_date)
            if existing.exists():
                return self._copy_to_dest(existing, dest_dir)

            if generate_if_missing:
                produced = self._generate_owp(
                    huc8=huc8,
                    user_dt=None,
                    return_period=return_period,
                    start_date=start_date,
                    end_date=end_date,
                )
                if produced and produced.exists():
                    return self._copy_to_dest(produced, dest_dir)
            return None

        # Standard Day/Hour Map
        if not user_dt:
            return None

        ymd, timestr = self._ymd_timestr_from_user(user_dt)

        if timestr is None:
            existing = self._find_any_owp_for_day(huc8, ymd)
        else:
            ep = self._expected_owp_path(huc8, ymd, timestr)
            existing = ep if ep.exists() else None

        if existing and existing.exists():
            return self._copy_to_dest(existing, dest_dir)

        if generate_if_missing:
            produced = self._generate_owp(huc8, user_dt, return_period=return_period)
            if produced and produced.exists():
                return self._copy_to_dest(produced, dest_dir)

        return None

    # OWP helpers
    def _ymd_timestr_from_user(self, user_dt: str) -> Tuple[str, Optional[str]]:
        day = _to_date(user_dt)
        hh = _to_hour_or_none(user_dt)
        ymd = day.strftime("%Y%m%d")
        return (ymd, None) if hh is None else (ymd, f"{hh:02d}0000")

    def _expected_owp_path(self, huc8: str, ymd: str, timestr: Optional[str]) -> Path:
        name = (
            f"NWM_{ymd}_{huc8}_inundation.tif"
            if timestr is None
            else f"NWM_{ymd}{timestr}_{huc8}_inundation.tif"
        )
        return self.owp_root / f"flood_{huc8}" / f"{huc8}_inundation" / name

    @staticmethod
    def _copy_to_dest(src: Path, dest_dir: str) -> str:
        dst_dir = Path(dest_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return str(dst)

    def _generate_owp(
        self,
        huc8: str,
        user_dt: Optional[str],
        return_period: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Main generation driver.
        - Supports HWM ranges with Maximum Discharge sort.
        - Supports return periods.
        - Supports standard event points.
        """
        self._ensure_roots()

        # Check existing first to avoid double processing
        if start_date and end_date:
            expected = self._expected_owp_path_hwm(huc8, start_date, end_date)
            if expected.exists():
                return expected
        elif user_dt:
            ymd, timestr = self._ymd_timestr_from_user(user_dt)
            expected = self._expected_owp_path(huc8, ymd, timestr)
            if expected.exists():
                return expected

        print(f"**Generating OWP HAND FIM for HUC {huc8}...**")
        DownloadHUC8(huc8, version="4.8")

        # HWM Range --> Maximum Discharge over the date range
        if start_date and end_date:
            print(
                f"Triggering NWM retrospective for range {start_date} to {end_date} (Sort: Maximum)"
            )
            getNWMretrospectivedata(
                huc=str(huc8),
                start_date=start_date,
                end_date=end_date,
                discharge_sortby="maximum",
            )
            runOWPHANDFIM(huc8)
            return self._expected_owp_path_hwm(huc8, start_date, end_date)

        # Return Period (BLE)--> get the flows from aws
        if return_period is not None:
            self._download_return_period_flows_csv(huc8, int(return_period))
            runOWPHANDFIM(huc8)
            return self._find_any_owp_for_return_period(huc8, int(return_period))

        # Standard specific event --> using the date input
        if user_dt:
            day = _to_date(user_dt)
            hh = _to_hour_or_none(user_dt)
            stamp = (
                f"{day:%Y-%m-%d}" if hh is None else f"{day:%Y-%m-%d} {hh:02d}:00:00"
            )
            getNWMretrospectivedata(huc_event_dict={str(huc8): [stamp]})
            runOWPHANDFIM(huc8)
            ymd, timestr = self._ymd_timestr_from_user(user_dt)
            return self._expected_owp_path(huc8, ymd, timestr)

        return None


# Main wrapper
def fim_lookup(
    HUCID: Optional[str] = None,
    date_input: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_period: Optional[int] = None,
    tier: Optional[str] = None,
    file_name: Optional[str] = None,
    run_handfim: bool = False,
    out_dir: Optional[str] = None,
    huc_intersectedarea: bool = False,
    huc_thresholdarea: float = 0.0,
    eval_individual_huc: bool = False,
) -> str:
    svc = FIMService()

    if file_name or run_handfim:
        rep = svc.process(
            huc8=HUCID,
            date_input=date_input,
            ensure_owp=run_handfim,
            generate_owp_if_missing=run_handfim,
            out_dir=out_dir,
            file_name=file_name,
            return_period=return_period,
            start_date=start_date,
            end_date=end_date,
            tier=tier,
            huc_thresholdarea=huc_thresholdarea,
            eval_individual_huc=eval_individual_huc,
        )
        return rep.get("message", "")

    # Query Mode Logic
    if not HUCID:
        return "Error: HUCID is required for query mode (listing available benchmarks)."

    q = svc.query(
        HUCID=HUCID,
        date_input=date_input,
        file_name=None,
        start_date=start_date,
        end_date=end_date,
        return_period=return_period,
        tier=tier,
        huc_intersectedarea=huc_intersectedarea,
        huc_thresholdarea=huc_thresholdarea,
    )

    txt = q.get("printable") or ""
    if not txt.strip():
        return f"No benchmark FIMs matched for HUC {HUCID}."

    return f"Following are the available benchmark data for HUC {HUCID}:\n{txt}"
