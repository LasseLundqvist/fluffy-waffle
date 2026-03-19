"""
Freight / shipping cost fetchers.

BDI (Baltic Dry Index): Stooq is currently unavailable; uses yfinance BDRY
(Breakwave Dry Bulk Shipping ETF) as a proxy for shipping cost conditions.

FAO Food Price Index: FAO FAOSTAT API is currently returning 404; falls back
to FRED PFOODINDEXM (requires API key) or yfinance agricultural futures.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from uk_cpi_nowcast.config import (
    FAO_BASE_URL,
    FRED_API_KEY,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def fetch_bdi(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch a Baltic Dry Index proxy (monthly).

    Tries in order:
    1. Stooq monthly BDI (direct)
    2. yfinance BDRY (Breakwave Dry Bulk Shipping ETF) as proxy

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, named "bdi".
    """
    # Try Stooq
    try:
        url = "https://stooq.com/q/d/l/?s=bdi&i=m"
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        from io import StringIO
        text = resp.text.strip()
        if text and text != "No data" and "Date" in text:
            df = pd.read_csv(StringIO(text))
            if not df.empty and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                val_col = "Close" if "Close" in df.columns else df.columns[0]
                s = df[val_col].resample("MS").mean()
                s = s[s.index >= pd.Timestamp(start_date)]
                s.name = "bdi"
                logger.info("BDI: fetched %d monthly observations from Stooq", len(s))
                return s
    except Exception as exc:
        logger.debug("BDI Stooq fetch failed: %s", exc)

    # Fallback: yfinance BDRY ETF
    try:
        import yfinance as yf
        raw = yf.download("BDRY", start=start_date, interval="1mo", progress=False, auto_adjust=True)
        if not raw.empty:
            s = raw["Close"].squeeze()
            s.index = pd.DatetimeIndex(
                [pd.Timestamp(d.year, d.month, 1) for d in pd.to_datetime(s.index)]
            )
            s = s[s.index >= pd.Timestamp(start_date)]
            s.name = "bdi"
            logger.info("BDI: using BDRY ETF proxy (%d obs) from yfinance", len(s))
            return s
    except Exception as exc:
        logger.warning("BDI yfinance BDRY fallback failed: %s", exc)

    logger.warning("BDI: all sources exhausted; returning empty series")
    return pd.Series(name="bdi", dtype=float)


def fetch_fao_food_price_index(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch a global food price index (monthly).

    Tries in order:
    1. FAO FAOSTAT API (Food Price Index)
    2. FRED PFOODINDEXM (requires FRED_API_KEY)
    3. yfinance agricultural futures basket (CORN + WHEAT + SOYB) as proxy

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, named "fao_food_price".
    """
    # FAO API
    url = f"{FAO_BASE_URL}/en/data/FP/preview"
    params = {
        "area_cs": "5000",
        "element_cs": "23013",
        "item_cs": "23013",
        "year": ",".join(str(y) for y in range(2010, 2026)),
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

            months_map = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12,
            }
            rows = []
            for rec in records:
                month_name = rec.get("months", "")
                month_num = months_map.get(month_name)
                year = rec.get("year")
                val = rec.get("value")
                if month_num and year and val is not None:
                    rows.append({
                        "date": pd.Timestamp(year=int(year), month=month_num, day=1),
                        "value": float(val),
                    })

            if rows:
                df = pd.DataFrame(rows).set_index("date").sort_index()
                s = df["value"]
                s = s[s.index >= pd.Timestamp(start_date)]
                s.name = "fao_food_price"
                logger.info("FAO: fetched %d monthly observations", len(s))
                return s

        except requests.HTTPError as exc:
            # 404 means the endpoint is gone — no point retrying
            if exc.response is not None and exc.response.status_code == 404:
                logger.debug("FAO API returned 404 — endpoint unavailable, skipping retries")
                break
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("FAO request failed (attempt %d/%d): %s.", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("FAO request failed (attempt %d/%d): %s.", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except Exception as exc:
            logger.debug("FAO parse error: %s", exc)
            break

    # FRED fallback (requires key)
    if FRED_API_KEY:
        try:
            from uk_cpi_nowcast.data.fetchers.fred import _fetch_fred_series
            s = _fetch_fred_series("PFOODINDEXM", start_date=start_date)
            if not s.empty:
                s.name = "fao_food_price"
                logger.info("FAO: using FRED PFOODINDEXM fallback")
                return s
        except Exception as exc:
            logger.warning("FRED food price fallback failed: %s", exc)

    # yfinance agricultural basket: average of CORN, WHEAT, SOYB (normalised)
    try:
        import yfinance as yf
        import numpy as np
        agri_tickers = {"corn": "ZC=F", "wheat": "ZW=F", "soybean": "ZS=F"}
        frames = {}
        for name, ticker in agri_tickers.items():
            raw = yf.download(ticker, start=start_date, interval="1mo",
                              progress=False, auto_adjust=True)
            if not raw.empty:
                s = raw["Close"].squeeze()
                s.index = pd.DatetimeIndex(
                [pd.Timestamp(d.year, d.month, 1) for d in pd.to_datetime(s.index)]
            )
                frames[name] = s

        if frames:
            df = pd.DataFrame(frames).dropna(how="all")
            # Normalise to 2015 base and average
            base = df.loc["2015-01-01":"2015-12-31"].mean()
            df_norm = df / base * 100
            basket = df_norm.mean(axis=1)
            basket = basket[basket.index >= pd.Timestamp(start_date)]
            basket.name = "fao_food_price"
            logger.info(
                "FAO: using yfinance agri-basket proxy (%d obs, tickers: %s)",
                len(basket), list(frames.keys()),
            )
            return basket
    except Exception as exc:
        logger.warning("yfinance agri-basket fallback failed: %s", exc)

    logger.warning("fao_food_price: all sources exhausted; returning empty series")
    return pd.Series(name="fao_food_price", dtype=float)
