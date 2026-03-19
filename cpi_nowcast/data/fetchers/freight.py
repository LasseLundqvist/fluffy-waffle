"""
Freight / shipping cost fetchers.

Baltic Dry Index (BDI) is fetched from FRED (available as a quarterly proxy)
or scraped from public sources.  FAO Food Price Index is fetched from the
FAO API.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from cpi_nowcast.config import (
    FAO_BASE_URL,
    FRED_API_KEY,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Quandl/FRED doesn't have BDI directly any more; use this public fallback
BDI_FALLBACK_URL = "https://markets.businessinsider.com/api/data/chart?isin=FR0000448074"


def fetch_bdi(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch the Baltic Dry Index (monthly average) from a public data source.

    Tries the FRED API first (for the old BDI series), then falls back to
    scraping a public financial data endpoint.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, BDI index level, named "bdi".
    """
    # FRED MABMM301DKM189S is actually M1 — BDI is not on FRED directly.
    # We use a web scraping fallback with the public data.
    try:
        # Try fetching from investing.com-style public API (Stooq)
        url = "https://stooq.com/q/d/l/?s=bdi&i=m"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        if df.empty or "Date" not in df.columns:
            raise ValueError("Unexpected BDI data format from Stooq")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # Stooq returns OHLCV — use Close
        val_col = "Close" if "Close" in df.columns else df.columns[0]
        s = df[val_col].resample("MS").mean()
        s = s[s.index >= pd.Timestamp(start_date)]
        s.name = "bdi"
        logger.info("BDI: fetched %d monthly observations from Stooq", len(s))
        return s
    except Exception as exc:
        logger.warning("BDI Stooq fetch failed: %s", exc)

    logger.error("BDI: all sources exhausted; returning empty series")
    return pd.Series(name="bdi", dtype=float)


def fetch_fao_food_price_index(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch the FAO Food Price Index (monthly) from the FAO FAOSTAT API.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, index (2014–2016 = 100), named "fao_food_price".
    """
    # FAO FAOSTAT bulk download — Food Price Index dataset
    url = f"{FAO_BASE_URL}/en/data/FP/preview"
    params = {
        "area_cs": "5000",      # World
        "element_cs": "23013",  # Food Price Index
        "item_cs": "23013",
        "year": "2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025",
        "output_type": "json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("data", [])
            if not records:
                raise ValueError("Empty FAO response")

            rows = []
            for rec in records:
                months_map = {
                    "January": 1, "February": 2, "March": 3, "April": 4,
                    "May": 5, "June": 6, "July": 7, "August": 8,
                    "September": 9, "October": 10, "November": 11, "December": 12,
                }
                month_name = rec.get("months", "")
                month_num = months_map.get(month_name, None)
                year = rec.get("year", None)
                val = rec.get("value", None)
                if month_num and year and val is not None:
                    rows.append({
                        "date": pd.Timestamp(year=int(year), month=month_num, day=1),
                        "value": float(val),
                    })

            if not rows:
                raise ValueError("No parseable rows in FAO response")

            df = pd.DataFrame(rows).set_index("date").sort_index()
            s = df["value"]
            s = s[s.index >= pd.Timestamp(start_date)]
            s.name = "fao_food_price"
            logger.info("FAO: fetched %d monthly Food Price Index observations", len(s))
            return s

        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "FAO request failed (attempt %d/%d): %s. Retrying in %ds…",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except Exception as exc:
            logger.warning("FAO parse error: %s", exc)
            break

    # Fallback: try FRED PFOODINDEXM (IMF commodity food price index, monthly)
    logger.info("FAO: trying FRED fallback for food price index")
    try:
        from cpi_nowcast.data.fetchers.fred import _fetch_fred_series
        s = _fetch_fred_series("PFOODINDEXM", start_date=start_date)
        if not s.empty:
            s.name = "fao_food_price"
            return s
    except Exception as exc:
        logger.warning("FRED food price fallback failed: %s", exc)

    logger.error("FAO food price: all sources exhausted; returning empty series")
    return pd.Series(name="fao_food_price", dtype=float)
