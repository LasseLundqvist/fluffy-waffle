"""
FRED data fetcher — oil prices, natural gas, USD/GBP, and fertilizer prices.

FRED requires an API key (set FRED_API_KEY env var).  When no key is available,
this module falls back to yfinance for commodity and FX series.

yfinance fallback tickers:
  brent_oil    → BZ=F  (Brent crude front-month futures, USD/barrel)
  natural_gas  → NG=F  (Henry Hub natural gas front-month futures)
  usd_gbp      → GBPUSD=X  (GBP/USD spot, inverted to USD/GBP)
  fertilizer   → DAP from yfinance is unavailable; skipped gracefully
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

import pandas as pd
import requests

from uk_cpi_nowcast.config import (
    FRED_API_KEY,
    FRED_SERIES,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fred_available() -> bool:
    """Return True if a FRED API key is configured."""
    return bool(FRED_API_KEY)


def _fetch_fred_series(
    series_id: str,
    start_date: str = HISTORY_START,
    end_date: Optional[str] = None,
    api_key: str = FRED_API_KEY,
) -> pd.Series:
    """
    Fetch a single FRED time series via the REST API.

    Returns an empty Series immediately if no API key is set.
    """
    if not api_key:
        logger.warning(
            "FRED: no API key set — skipping %s. "
            "Set FRED_API_KEY env var to enable FRED data.",
            series_id,
        )
        return pd.Series(name=series_id, dtype=float)

    if end_date is None:
        end_date = date.today().isoformat()

    params: dict = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "file_type": "json",
        "units": "lin",
        "api_key": api_key,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=REQUEST_TIMEOUT)
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


def _fetch_yfinance(
    ticker: str,
    name: str,
    start_date: str = HISTORY_START,
    invert: bool = False,
) -> pd.Series:
    """
    Fetch monthly series from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker, e.g. "BZ=F".
    name : str
        Output series name.
    start_date : str
        Start date for the series.
    invert : bool
        If True, return 1/value (e.g. convert GBP/USD to USD/GBP).
    """
    try:
        import yfinance as yf
        raw = yf.download(
            ticker,
            start=start_date,
            interval="1mo",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            logger.warning("yfinance: empty result for %s (%s)", name, ticker)
            return pd.Series(name=name, dtype=float)

        # yfinance monthly returns start-of-month dates
        close = raw["Close"].squeeze()
        if invert:
            close = 1.0 / close
        # Normalise index to month-start (yfinance monthly returns first day of month)
        close.index = pd.DatetimeIndex(
            [pd.Timestamp(d.year, d.month, 1) for d in pd.to_datetime(close.index)]
        )
        close = close[close.index >= pd.Timestamp(start_date)]
        close.name = name
        logger.info("yfinance: fetched %d monthly obs for %s (%s)", len(close), name, ticker)
        return close
    except ImportError:
        logger.warning("yfinance not installed; cannot fetch %s. Run: pip install yfinance", name)
        return pd.Series(name=name, dtype=float)
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s (%s): %s", name, ticker, exc)
        return pd.Series(name=name, dtype=float)


def fetch_brent_oil(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch daily Brent crude oil prices (USD/barrel).

    Primary: FRED DCOILBRENTEU (requires API key).
    Fallback: yfinance BZ=F (Brent front-month futures, monthly).
    """
    if _fred_available():
        series = _fetch_fred_series(FRED_SERIES["brent_oil"], start_date=start_date)
        if not series.empty:
            series.name = "brent_oil"
            return series

    logger.info("fetch_brent_oil: using yfinance BZ=F fallback")
    return _fetch_yfinance("BZ=F", "brent_oil", start_date=start_date)


def fetch_natural_gas(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch natural gas prices.

    Primary: FRED PNGASEUUSDM TTF (requires API key).
    Fallback: yfinance NG=F (Henry Hub front-month futures, monthly).
    """
    if _fred_available():
        series = _fetch_fred_series(FRED_SERIES["natural_gas_ttf"], start_date=start_date)
        if not series.empty:
            series.name = "natural_gas"
            return series

    logger.info("fetch_natural_gas: using yfinance NG=F fallback")
    return _fetch_yfinance("NG=F", "natural_gas", start_date=start_date)


def fetch_usd_gbp(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch USD per GBP exchange rate.

    Primary: FRED DEXUSUK (requires API key).
    Fallback: yfinance GBPUSD=X (GBP/USD, inverted to USD/GBP).

    Note: The ECB fetcher also computes USD/GBP as a cross-rate;
    this provides a direct daily series.
    """
    if _fred_available():
        series = _fetch_fred_series(FRED_SERIES["usd_gbp"], start_date=start_date)
        if not series.empty:
            series.name = "usd_gbp"
            return series

    logger.info("fetch_usd_gbp: using yfinance GBPUSD=X fallback (inverted)")
    return _fetch_yfinance("GBPUSD=X", "usd_gbp", start_date=start_date, invert=True)


def fetch_fertilizer_prices(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch IMF Fertilizer Price Index (monthly).

    Primary: FRED PFERTILIZERINDEXM (requires API key).
    Fallback: FRED PUREAM (Urea price, also requires API key).
    No public free alternative available; returns empty if no API key.
    """
    if _fred_available():
        series = _fetch_fred_series(FRED_SERIES["fertilizer"], start_date=start_date)
        if not series.empty:
            series.name = "fertilizer"
            return series
        # Urea fallback
        series = _fetch_fred_series("PUREAM", start_date=start_date)
        if not series.empty:
            series.name = "fertilizer"
            return series

    logger.warning(
        "fetch_fertilizer_prices: no FRED API key — set FRED_API_KEY to include fertilizer data"
    )
    return pd.Series(name="fertilizer", dtype=float)
