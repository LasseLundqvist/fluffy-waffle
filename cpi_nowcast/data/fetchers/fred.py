"""
FRED data fetcher — oil prices, natural gas, and other commodity series.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

import pandas as pd
import requests

from cpi_nowcast.config import (
    FRED_API_KEY,
    FRED_SERIES,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_fred_series(
    series_id: str,
    start_date: str = HISTORY_START,
    end_date: Optional[str] = None,
    api_key: str = FRED_API_KEY,
) -> pd.Series:
    """
    Fetch a single FRED time series via the REST API.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. "DCOILBRENTEU".
    start_date : str
        ISO date string for the first observation.
    end_date : str, optional
        ISO date string for the last observation.  Defaults to today.
    api_key : str
        FRED API key.  Falls back to public (rate-limited) endpoint if empty.

    Returns
    -------
    pd.Series
        DatetimeIndex, float values, named after series_id.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    params: dict = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "file_type": "json",
        "units": "lin",
    }
    if api_key:
        params["api_key"] = api_key

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                FRED_API_URL, params=params, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            if not observations:
                logger.warning("FRED: no observations returned for %s", series_id)
                return pd.Series(name=series_id, dtype=float)

            records = {
                obs["date"]: float(obs["value"])
                for obs in observations
                if obs["value"] != "."
            }
            series = pd.Series(records, name=series_id)
            series.index = pd.to_datetime(series.index)
            series = series.sort_index()
            logger.info("FRED: fetched %d obs for %s", len(series), series_id)
            return series

        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "FRED request failed (attempt %d/%d): %s. Retrying in %ds…",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    logger.error("FRED: all retries exhausted for %s", series_id)
    return pd.Series(name=series_id, dtype=float)


def fetch_brent_oil(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch daily Brent crude oil prices (USD/barrel) from FRED.

    Returns
    -------
    pd.Series
        Daily series named "brent_oil".
    """
    series = _fetch_fred_series(FRED_SERIES["brent_oil"], start_date=start_date)
    series.name = "brent_oil"
    return series


def fetch_natural_gas(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch monthly natural gas (TTF proxy) prices from FRED.

    Returns
    -------
    pd.Series
        Monthly series named "natural_gas".
    """
    series = _fetch_fred_series(
        FRED_SERIES["natural_gas_ttf"], start_date=start_date
    )
    series.name = "natural_gas"
    return series


def fetch_all_fred(start_date: str = HISTORY_START) -> pd.DataFrame:
    """
    Fetch all configured FRED series and return them as a wide DataFrame.

    Daily series are kept at daily frequency; monthly series are kept monthly.
    Merging across frequencies is handled by the pipeline.

    Returns
    -------
    pd.DataFrame
        Columns are series names; index is DatetimeIndex.
    """
    frames: list[pd.Series] = []
    for name, series_id in FRED_SERIES.items():
        s = _fetch_fred_series(series_id, start_date=start_date)
        s.name = name
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    return df
