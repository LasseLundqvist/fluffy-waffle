"""
Danmarks Statistik (DST) API fetcher.

Fetches CPI/HICP target variable and consumer confidence data.
Uses the Statistics Denmark JSON API v1.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests

from cpi_nowcast.config import (
    DST_BASE_URL,
    DST_TABLES,
    HISTORY_START,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _dst_post(
    table: str,
    variables: list[dict[str, Any]],
    start_date: str = HISTORY_START,
) -> pd.DataFrame:
    """
    POST a query to the DST JSON API and return a tidy DataFrame.

    Parameters
    ----------
    table : str
        DST table name, e.g. "PRIS111".
    variables : list of dict
        List of variable filter dicts accepted by the DST API.
    start_date : str
        ISO date string; observations before this are dropped.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: period, value, and any group columns.
    """
    url = f"{DST_BASE_URL}/data"
    payload: dict[str, Any] = {
        "table": table,
        "format": "JSON",
        "lang": "en",
        "variables": variables,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            columns = [h["id"] for h in data.get("head", {}).get("vars", [])]
            rows = data.get("data", [])
            if not rows:
                logger.warning("DST: no data returned for table %s", table)
                return pd.DataFrame(columns=columns + ["value"])

            records = []
            for row in rows:
                record = dict(zip(columns, row["key"]))
                record["value"] = float(row["values"][0]) if row["values"] else None
                records.append(record)

            df = pd.DataFrame(records)
            return df

        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "DST request failed (attempt %d/%d): %s. Retrying in %ds…",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    logger.error("DST: all retries exhausted for table %s", table)
    return pd.DataFrame()


def fetch_cpi(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch the Danish national CPI (all items, total) from DST PRIS111.

    The index reflects total consumer prices, seasonally unadjusted.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, CPI index level (2015=100), named "cpi".
    """
    variables = [
        {"code": "PRISTYPE", "values": ["KPITOTALE"]},
        {"code": "ENHED",    "values": ["INDEKS"]},
        {"code": "Tid",      "values": ["*"]},
    ]
    df = _dst_post(DST_TABLES["cpi"], variables, start_date=start_date)
    if df.empty:
        logger.warning("DST CPI: empty response, returning empty series")
        return pd.Series(name="cpi", dtype=float)

    # Period column is typically named "TID" or "Tid"
    period_col = next(
        (c for c in df.columns if c.upper() in ("TID", "PERIOD")), None
    )
    if period_col is None and "value" not in df.columns:
        logger.error("DST CPI: could not identify period column. Columns: %s", df.columns.tolist())
        return pd.Series(name="cpi", dtype=float)
    if period_col is None:
        period_col = df.columns[0]

    df = df[[period_col, "value"]].dropna(subset=["value"])
    # DST period format: "2023M01"
    df[period_col] = df[period_col].astype(str).str.replace(
        r"(\d{4})M(\d{2})", r"\1-\2", regex=True
    )
    df[period_col] = pd.to_datetime(df[period_col], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=[period_col])
    df = df[df[period_col] >= pd.Timestamp(start_date)]
    series = df.set_index(period_col)["value"].sort_index()
    series.index.freq = pd.tseries.frequencies.to_offset("MS")
    series.name = "cpi"
    logger.info("DST: fetched %d CPI observations", len(series))
    return series


def fetch_consumer_confidence(start_date: str = HISTORY_START) -> pd.Series:
    """
    Fetch the consumer confidence price expectations component from DST FORV1.

    Returns
    -------
    pd.Series
        Monthly DatetimeIndex, named "consumer_conf_prices".
    """
    variables = [
        {"code": "INDIKATOR", "values": ["PFOR"]},   # Price expectations next 12M
        {"code": "Tid",       "values": ["*"]},
    ]
    df = _dst_post(DST_TABLES["consumer_conf"], variables, start_date=start_date)
    if df.empty:
        logger.warning("DST consumer confidence: empty response")
        return pd.Series(name="consumer_conf_prices", dtype=float)

    period_col = next(
        (c for c in df.columns if c.upper() in ("TID", "PERIOD")), df.columns[0]
    )
    df = df[[period_col, "value"]].dropna(subset=["value"])
    df[period_col] = df[period_col].astype(str).str.replace(
        r"(\d{4})M(\d{2})", r"\1-\2", regex=True
    )
    df[period_col] = pd.to_datetime(df[period_col], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=[period_col])
    df = df[df[period_col] >= pd.Timestamp(start_date)]
    series = df.set_index(period_col)["value"].sort_index()
    series.name = "consumer_conf_prices"
    logger.info("DST: fetched %d consumer confidence observations", len(series))
    return series


def fetch_all_dst(start_date: str = HISTORY_START) -> pd.DataFrame:
    """
    Fetch all DST series and return as a combined monthly DataFrame.

    Returns
    -------
    pd.DataFrame
        Monthly DatetimeIndex, columns: cpi, consumer_conf_prices.
    """
    cpi = fetch_cpi(start_date=start_date)
    conf = fetch_consumer_confidence(start_date=start_date)
    return pd.concat([cpi, conf], axis=1)
