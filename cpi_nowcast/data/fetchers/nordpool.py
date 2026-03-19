"""
Nord Pool / ENTSO-E electricity price fetcher for Denmark (DK1 and DK2).

Primary source: ENTSO-E Transparency Platform REST API.
Fallback: simple web scraping of public Nord Pool data.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from cpi_nowcast.config import (
    ENTSO_E_BASE_URL,
    ENTSO_E_TOKEN,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ENTSO-E area codes
AREA_DK1 = "10YDK-1--------W"
AREA_DK2 = "10YDK-2--------M"

# Fallback: Nord Pool public data (no auth needed, aggregated monthly)
NORDPOOL_PUBLIC_URL = (
    "https://www.nordpoolgroup.com/api/marketdata/page/10"
)


def _fetch_entsoe_prices(
    area_eic: str,
    start_date: str,
    end_date: str,
    token: str,
) -> pd.Series:
    """
    Fetch hourly day-ahead prices from ENTSO-E Transparency Platform.

    Parameters
    ----------
    area_eic : str
        EIC bidding zone code.
    start_date : str
        ISO date string.
    end_date : str
        ISO date string.
    token : str
        ENTSO-E security token.

    Returns
    -------
    pd.Series
        Hourly prices (EUR/MWh).
    """
    # ENTSO-E requires YYYYMMDD format
    s_fmt = pd.Timestamp(start_date).strftime("%Y%m%d%H%M")
    e_fmt = pd.Timestamp(end_date).strftime("%Y%m%d%H%M")

    params = {
        "securityToken": token,
        "documentType": "A44",        # Price document
        "in_Domain": area_eic,
        "out_Domain": area_eic,
        "periodStart": s_fmt,
        "periodEnd": e_fmt,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                ENTSO_E_BASE_URL, params=params, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            # Parse the XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

            records: list[tuple[pd.Timestamp, float]] = []
            for ts_elem in root.findall(".//ns:TimeSeries", ns):
                for period in ts_elem.findall(".//ns:Period", ns):
                    start_el = period.find(".//ns:timeInterval/ns:start", ns)
                    if start_el is None:
                        continue
                    period_start = pd.Timestamp(start_el.text)
                    resolution = period.find(".//ns:resolution", ns)
                    freq_minutes = 60  # default hourly
                    if resolution is not None and resolution.text:
                        if "PT60M" in resolution.text:
                            freq_minutes = 60
                        elif "PT15M" in resolution.text:
                            freq_minutes = 15
                    for pt in period.findall(".//ns:Point", ns):
                        pos_el = pt.find("ns:position", ns)
                        val_el = pt.find("ns:price.amount", ns)
                        if pos_el is not None and val_el is not None:
                            pos = int(pos_el.text) - 1
                            ts = period_start + pd.Timedelta(minutes=pos * freq_minutes)
                            records.append((ts, float(val_el.text)))

            if not records:
                return pd.Series(dtype=float)
            idx, vals = zip(*records)
            return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()

        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "ENTSO-E request failed (attempt %d/%d): %s. Retrying in %ds…",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    return pd.Series(dtype=float)


def _fetch_nordpool_fallback(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scrape monthly average spot prices from Nord Pool public API.

    Returns
    -------
    pd.DataFrame
        Columns: dk1_eur_mwh, dk2_eur_mwh; monthly DatetimeIndex.
    """
    try:
        resp = requests.get(NORDPOOL_PUBLIC_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", {}).get("Rows", [])
        records = []
        for row in rows:
            try:
                month_str = row.get("Name", "")
                if not month_str:
                    continue
                cols = row.get("Columns", [])
                dk1 = next(
                    (
                        float(c["Value"].replace(",", ".").replace("\xa0", ""))
                        for c in cols
                        if c.get("Name", "").upper() in ("DK1", "DK 1")
                    ),
                    None,
                )
                dk2 = next(
                    (
                        float(c["Value"].replace(",", ".").replace("\xa0", ""))
                        for c in cols
                        if c.get("Name", "").upper() in ("DK2", "DK 2")
                    ),
                    None,
                )
                records.append({"period": month_str, "dk1": dk1, "dk2": dk2})
            except (ValueError, AttributeError):
                continue

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame()
        df["period"] = pd.to_datetime(df["period"], errors="coerce")
        df = df.dropna(subset=["period"])
        df = df.set_index("period").sort_index()
        df = df.rename(columns={"dk1": "dk1_eur_mwh", "dk2": "dk2_eur_mwh"})
        df = df[df.index >= pd.Timestamp(start_date)]
        return df
    except Exception as exc:
        logger.warning("Nord Pool fallback failed: %s", exc)
        return pd.DataFrame()


def fetch_elspot_prices(
    start_date: str = HISTORY_START,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch Danish electricity spot prices (DK1 and DK2).

    Tries ENTSO-E first (requires token); falls back to Nord Pool public API.
    Hourly prices are aggregated to monthly means.

    Returns
    -------
    pd.DataFrame
        Monthly DatetimeIndex; columns dk1_eur_mwh, dk2_eur_mwh.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    if ENTSO_E_TOKEN:
        logger.info("Fetching electricity prices from ENTSO-E")
        dk1_hourly = _fetch_entsoe_prices(
            AREA_DK1, start_date, end_date, ENTSO_E_TOKEN
        )
        dk2_hourly = _fetch_entsoe_prices(
            AREA_DK2, start_date, end_date, ENTSO_E_TOKEN
        )
        if not dk1_hourly.empty or not dk2_hourly.empty:
            df = pd.concat(
                [dk1_hourly.rename("dk1_eur_mwh"), dk2_hourly.rename("dk2_eur_mwh")],
                axis=1,
            )
            # Resample to monthly mean
            df_monthly = df.resample("MS").mean()
            return df_monthly

    # Fallback
    logger.info("Falling back to Nord Pool public data for electricity prices")
    return _fetch_nordpool_fallback(start_date, end_date)
