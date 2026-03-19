"""
ONS (Office for National Statistics) API fetcher — UK CPI/CPIH data.

Used as the primary source for UK CPI when Eurostat data is unavailable or stale.

Working endpoint (confirmed 2026-03):
    GET https://api.ons.gov.uk/v1/data?uri=/economy/inflationandpriceindices/timeseries/{id}/{dataset}

Series URIs:
    D7BT / MM23  — CPI All Items Index (2015=100)
    L55O / MM23  — CPIH All Items Index (2015=100)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from uk_cpi_nowcast.config import (
    ONS_BASE_URL,
    ONS_DATASET,
    ONS_SERIES,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

# URI paths for each ONS series (confirmed working)
_ONS_URIS = {
    "D7BT": "/economy/inflationandpriceindices/timeseries/d7bt/mm23",
    "L55O": "/economy/inflationandpriceindices/timeseries/l55o/mm23",
}


def _fetch_ons_timeseries(
    timeseries_id: str,
    dataset_id: str = ONS_DATASET,
) -> pd.DataFrame:
    """
    Fetch monthly timeseries data from the ONS API.

    Uses the /v1/data?uri= endpoint which returns a JSON object with a
    'months' array, each item having 'date' (e.g. "2015 JAN") and 'value'.

    Parameters
    ----------
    timeseries_id : str
        ONS timeseries code, e.g. "D7BT".
    dataset_id : str
        ONS dataset code, e.g. "MM23".

    Returns
    -------
    pd.DataFrame
        DataFrame with 'date' (Timestamp) and 'value' (float) columns.
    """
    uri = _ONS_URIS.get(
        timeseries_id.upper(),
        f"/economy/inflationandpriceindices/timeseries/{timeseries_id.lower()}/{dataset_id.lower()}",
    )
    url = f"{ONS_BASE_URL}/data"
    params = {"uri": uri}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            months = data.get("months", [])
            if not months:
                logger.warning("ONS: no monthly data for %s/%s", timeseries_id, dataset_id)
                return pd.DataFrame()

            rows = []
            for item in months:
                date_str = item.get("date", "")   # "YYYY MMM" e.g. "2015 JAN"
                value_str = item.get("value", "")
                if date_str and value_str:
                    try:
                        dt = pd.to_datetime(date_str, format="%Y %b")
                        rows.append({"date": dt, "value": float(value_str)})
                    except (ValueError, TypeError):
                        continue

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            logger.info(
                "ONS: fetched %d monthly observations for %s", len(df), timeseries_id
            )
            return df

        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "ONS request failed (attempt %d/%d): %s. Retrying in %ds…",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    logger.error("ONS: all retries exhausted for %s/%s", timeseries_id, dataset_id)
    return pd.DataFrame()


def fetch_uk_cpi(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch UK CPI All Items Index (2015=100) from ONS.

    This is the primary fallback when Eurostat UK HICP data is unavailable.
    Series D7BT from dataset MM23.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, named "cpi_uk".
    """
    df = _fetch_ons_timeseries(ONS_SERIES["cpi_uk"])
    if df.empty:
        return pd.Series(name="cpi_uk", dtype=float)

    s = df.set_index("date")["value"]
    s = s[s.index >= pd.Timestamp(start_date)]
    s.name = "cpi_uk"
    return s


def fetch_uk_cpih(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch UK CPIH All Items Index (2015=100) from ONS.

    CPIH includes owner-occupiers' housing costs (OOH) and is the ONS's
    headline measure.  Used as an additional feature.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, named "cpih_uk".
    """
    df = _fetch_ons_timeseries(ONS_SERIES["cpih_uk"])
    if df.empty:
        return pd.Series(name="cpih_uk", dtype=float)

    s = df.set_index("date")["value"]
    s = s[s.index >= pd.Timestamp(start_date)]
    s.name = "cpih_uk"
    return s
