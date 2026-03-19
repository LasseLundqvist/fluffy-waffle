"""
Eurostat SDMX fetcher — UK HICP (target series) and EA HICP flash (indicator).

The UK HICP from Eurostat (dataset prc_hicp_midx, country code "UK") is the
primary target.  The euro-area flash estimate is included as a leading indicator
since it is released ~30 days before the final figure and is correlated with UK
inflation via trade and common global shocks.
"""
from __future__ import annotations

import logging
import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests

from uk_cpi_nowcast.config import (
    EUROSTAT_BASE_URL,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _fetch_eurostat_sdmx(
    dataset: str,
    filter_expr: str,
    start_period: str = "2010",
) -> pd.DataFrame:
    """
    Fetch a Eurostat dataset via the SDMX 2.1 REST API (TSV format).

    The Eurostat SDMX 2.1 API returns wide-format TSV: metadata key as the
    first column, date periods as remaining column headers.

    Parameters
    ----------
    dataset : str
        Eurostat dataset code, e.g. "prc_hicp_midx".
    filter_expr : str
        SDMX filter, e.g. "M.I15.CP00.UK".
    start_period : str
        Start period in SDMX notation (e.g. "2010" or "2010-01").

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with TIME_PERIOD and OBS_VALUE columns (long format).
    """
    url = f"{EUROSTAT_BASE_URL}/data/{dataset}/{filter_expr}"
    params = {
        "format": "TSV",    # TSV is accepted; 'csvdata' returns 406; lang=EN causes 400 with TSV
        "startPeriod": start_period,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            # Parse wide-format TSV → long DataFrame
            df = _parse_eurostat_tsv(resp.text)
            logger.info(
                "Eurostat: fetched %d observations for %s/%s",
                len(df), dataset, filter_expr,
            )
            return df
        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "Eurostat request failed (attempt %d/%d): %s. Retrying in %ds…",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    logger.error("Eurostat: all retries exhausted for %s/%s", dataset, filter_expr)
    return pd.DataFrame()


def _parse_eurostat_tsv(text: str) -> pd.DataFrame:
    """
    Parse Eurostat TSV wide-format response into a long DataFrame.

    TSV layout example:
        freq,unit,coicop,geo\\TIME_PERIOD  2010-01   2010-02  ...
        M,I15,CP00,UK                       87.8      88.2    ...

    Values may contain flags like "87.8 b" or ": p" — numeric part is extracted.

    Returns
    -------
    pd.DataFrame
        Columns: TIME_PERIOD (str), OBS_VALUE (float).
    """
    import numpy as np

    lines = text.strip().splitlines()
    if not lines:
        return pd.DataFrame()

    # Header: first field is metadata key, remaining are date strings
    header_parts = lines[0].split("\t")
    date_cols = [d.strip() for d in header_parts[1:]]

    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        for date_str, val_str in zip(date_cols, parts[1:]):
            date_str = date_str.strip()
            raw = val_str.strip()
            if not raw or raw in (":", "n.a.", "N/A"):
                continue
            # Extract numeric part (ignore flag suffixes like "b", "p", "e")
            numeric = raw.split(":")[0].strip().split(" ")[0]
            try:
                obs_value = float(numeric)
            except ValueError:
                continue
            rows.append({"TIME_PERIOD": date_str, "OBS_VALUE": obs_value})

    return pd.DataFrame(rows)


def _parse_eurostat_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Parse TIME_PERIOD / OBS_VALUE columns from Eurostat CSV into a Series."""
    time_col = next(
        (c for c in df.columns if c.upper() in ("TIME_PERIOD", "TIME")), None
    )
    val_col = next(
        (c for c in df.columns if c.upper() in ("OBS_VALUE", "VALUE")), None
    )
    if time_col is None or val_col is None:
        logger.warning("Eurostat: unexpected columns %s for %s", df.columns.tolist(), name)
        return pd.Series(name=name, dtype=float)

    s = df.set_index(time_col)[val_col].astype(float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = name
    return s


def fetch_uk_hicp(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch UK HICP monthly index from Eurostat (country code "UK").

    Dataset: prc_hicp_midx — monthly index, all items (CP00), 2015=100.
    This is the primary target variable for UK CPI nowcasting.

    Note: Eurostat uses "UK" as the country code for the United Kingdom.
    Data availability may lag after Brexit; the ONS fetcher provides a fallback.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, named "hicp_uk".
    """
    start_year = pd.Timestamp(start_date).year
    filter_expr = "M.I15.CP00.UK"
    df = _fetch_eurostat_sdmx("prc_hicp_midx", filter_expr, start_period=str(start_year))

    if df.empty:
        logger.warning("Eurostat UK HICP returned empty — will use ONS fallback")
        return pd.Series(name="hicp_uk", dtype=float)

    s = _parse_eurostat_series(df, "hicp_uk")
    s = s[s.index >= pd.Timestamp(start_date)]
    return s


def fetch_ea_hicp_flash(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch EA HICP flash estimate from Eurostat.

    The flash estimate is released ~30 days before the final figure and is a
    useful leading indicator for UK inflation via common global commodity and
    trade channels.

    Returns
    -------
    pd.Series
        Monthly series named "hicp_flash_ea".
    """
    start_year = pd.Timestamp(start_date).year
    filter_expr = "M.I15.CP00.EA"
    df = _fetch_eurostat_sdmx("prc_hicp_midx", filter_expr, start_period=str(start_year))

    if df.empty:
        return pd.Series(name="hicp_flash_ea", dtype=float)

    s = _parse_eurostat_series(df, "hicp_flash_ea")
    s = s[s.index >= pd.Timestamp(start_date)]
    return s
