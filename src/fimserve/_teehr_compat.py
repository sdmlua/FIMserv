"""Compatibility shim for teehr 0.6.6 against current pandera/pandas.

teehr writes its parquet output through a pandera schema whose datetime columns
run the ``format_datetime64`` parser. That parser calls ``s.dt`` directly, but
teehr feeds it an all-NaN ``reference_time`` column (a float Series), and newer
pandera runs parsers before coercion. ``.dt`` on a non-datetime Series raises
"Can only use .dt accessor with datetimelike values", which aborts every
retrospective and USGS download.

We replace the parser with one that coerces to datetime first and only strips a
timezone when one is present. The schema builders look the name up as a module
global at call time, so patching the module attribute is enough.
"""
import pandas as pd


def _format_datetime64(s: pd.Series) -> pd.Series:
    """Coerce a Series to tz-naive datetime64[ms], tolerating non-datetime input."""
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.astype("datetime64[ms]")


def apply() -> None:
    """Patch teehr's datetime parser in place. Safe to call more than once."""
    try:
        from teehr.models import pandera_dataframe_schemas as schemas
    except Exception:
        return
    schemas.format_datetime64 = _format_datetime64
