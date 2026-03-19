"""
ECB SDMX API fetcher — EUR/GBP exchange rate and EUR OIS/swap rates.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from uk_cpi_nowcast.config import (
    ECB_BASE_URL,
    ECB_SERIES,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _fetch_ecb_series(
    series_key: str,
    start_date: str = HISTORY_START,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Fetch a single time series from the ECB SDMX 2.1 REST API.

    Parameters
    ----------
    series_key : str
        Full SDMX series key, e.g. "EXR.D.GBP.EUR.SP00.A".
    start_date : str
        ISO date string (YYYY-MM-DD).
    end_date : str, optional
        ISO date string.

    Returns
    -------
    pd.Series
        DatetimeIndex, float values, named after series_key.
    """
    flow_ref, key = series_key.split(".", 1)
    url = f"{ECB_BASE_URL}/data/{flow_ref}/{key}"
    params: dict = {
        "format": "csvdata",
        "startPeriod": start_date,
    }
    if end_date:
        params["endPeriod"] = end_date

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
                logger.warning(
                    "ECB: unexpected CSV columns for %s: %s",
                    series_key, df.columns.tolist()
                )
                return pd.Series(name=series_key, dtype=float)

            series = df.set_index("TIME_PERIOD")["OBS_VALUE"]
            series.index = pd.to_datetime(series.index)
            series = series.sort_index().astype(float)
            series.name = series_key
            logger.info("ECB: fetched %d obs for %s", len(series), series_key)
            return series

        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "ECB request failed (attempt %d/%d): %s. Retrying in %ds…",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    logger.error("ECB: all retries exhausted for %s", series_key)
    return pd.Series(name=series_key, dtype=float)


def fetch_eur_gbp(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch daily EUR/GBP exchange rate from ECB.

    The ECB series EXR.D.GBP.EUR.SP00.A gives GBP per EUR (spot rate).

    Returns
    -------
    pd.Series
        Daily series named "eur_gbp".
    """
    s = _fetch_ecb_series(ECB_SERIES["eur_gbp"], start_date=start_date)
    s.name = "eur_gbp"
    return s


def fetch_usd_gbp_cross(start_date: str = HISTORY_START) -> pd.Series:
    """
    Compute daily USD/GBP cross rate from ECB EUR exchange rates.

    USD/GBP = (USD per EUR) / (GBP per EUR)

    This avoids requiring a FRED API key for the USD/GBP rate.

    Returns
    -------
    pd.Series
        Daily series named "usd_gbp_ecb" (USD per GBP).
    """
    usd_eur = _fetch_ecb_series(ECB_SERIES["usd_eur"], start_date=start_date)
    eur_gbp = _fetch_ecb_series(ECB_SERIES["eur_gbp"], start_date=start_date)

    if usd_eur.empty or eur_gbp.empty:
        logger.warning("ECB: cannot compute USD/GBP cross rate — missing USD/EUR or EUR/GBP")
        return pd.Series(name="usd_gbp_ecb", dtype=float)

    # Align on common dates
    common = usd_eur.index.intersection(eur_gbp.index)
    cross = usd_eur.loc[common] / eur_gbp.loc[common]
    cross.name = "usd_gbp_ecb"
    logger.info("ECB: computed USD/GBP cross rate (%d obs)", len(cross))
    return cross


def fetch_ois_rates(start_date: str = HISTORY_START) -> pd.DataFrame:
    """
    Fetch ECB EUR OIS/swap rates for 2Y and 5Y tenors.

    These capture the euro-area risk-free rate expectations and are used as
    a proxy for global interest rate conditions affecting UK inflation via
    imported goods and financial conditions.

    Returns
    -------
    pd.DataFrame
        Columns: ois_2y, ois_5y.
    """
    frames: list[pd.Series] = []
    for name in ("ois_2y", "ois_5y"):
        s = _fetch_ecb_series(ECB_SERIES[name], start_date=start_date)
        s.name = name
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)
