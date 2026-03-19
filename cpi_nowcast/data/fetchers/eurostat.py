"""
Eurostat SDMX fetcher — HICP flash estimate and EU oil bulletin prices.
"""
from __future__ import annotations

import logging
import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests

from cpi_nowcast.config import (
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
    Fetch a Eurostat dataset via the SDMX 2.1 REST API (CSV format).

    Parameters
    ----------
    dataset : str
        Eurostat dataset code, e.g. "prc_hicp_midx".
    filter_expr : str
        SDMX filter, e.g. "M.I05.CP00.DK".
    start_period : str
        Start period in SDMX notation (e.g. "2010" or "2010-01").

    Returns
    -------
    pd.DataFrame
        Raw CSV parsed to DataFrame.
    """
    url = f"{EUROSTAT_BASE_URL}/data/{dataset}/{filter_expr}"
    params = {
        "format": "csvdata",
        "startPeriod": start_period,
        "lang": "EN",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            logger.info(
                "Eurostat: fetched %d rows for %s/%s", len(df), dataset, filter_expr
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


def fetch_hicp_flash(
    country: str = "EA",
    start_date: str = HISTORY_START,
) -> pd.Series:
    """
    Fetch the HICP flash estimate for the euro area (or a specific country).

    The flash estimate is released ~30 days before the final figure,
    making it a valuable leading indicator.

    Parameters
    ----------
    country : str
        Eurostat country/area code. "EA" = euro area.
    start_date : str
        ISO date string.

    Returns
    -------
    pd.Series
        Monthly series of HICP flash index (2015=100), named "hicp_flash_{country}".
    """
    start_year = pd.Timestamp(start_date).year
    # Dataset: prc_hicp_midx — monthly index, all items, index 2015=100
    filter_expr = f"M.I15.CP00.{country}"
    df = _fetch_eurostat_sdmx("prc_hicp_midx", filter_expr, start_period=str(start_year))

    if df.empty:
        return pd.Series(name=f"hicp_flash_{country.lower()}", dtype=float)

    time_col = next(
        (c for c in df.columns if c.upper() in ("TIME_PERIOD", "TIME")), None
    )
    val_col = next(
        (c for c in df.columns if c.upper() in ("OBS_VALUE", "VALUE")), None
    )

    if time_col is None or val_col is None:
        logger.warning(
            "Eurostat HICP: unexpected columns %s", df.columns.tolist()
        )
        return pd.Series(name=f"hicp_flash_{country.lower()}", dtype=float)

    s = df.set_index(time_col)[val_col].astype(float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[s.index >= pd.Timestamp(start_date)]
    s.name = f"hicp_flash_{country.lower()}"
    return s


def fetch_dk_hicp(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch Danish HICP monthly index from Eurostat.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, named "hicp_dk".
    """
    start_year = pd.Timestamp(start_date).year
    filter_expr = "M.I15.CP00.DK"
    df = _fetch_eurostat_sdmx("prc_hicp_midx", filter_expr, start_period=str(start_year))

    if df.empty:
        return pd.Series(name="hicp_dk", dtype=float)

    time_col = next(
        (c for c in df.columns if c.upper() in ("TIME_PERIOD", "TIME")), None
    )
    val_col = next(
        (c for c in df.columns if c.upper() in ("OBS_VALUE", "VALUE")), None
    )

    if time_col is None or val_col is None:
        return pd.Series(name="hicp_dk", dtype=float)

    s = df.set_index(time_col)[val_col].astype(float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[s.index >= pd.Timestamp(start_date)]
    s.name = "hicp_dk"
    return s
