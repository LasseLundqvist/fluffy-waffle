"""
Data pipeline: orchestrates all fetchers, preprocesses data, caches to parquet.

Key responsibilities:
- Fetch and merge all data sources into a single wide DataFrame
- Convert to year-over-year growth rates
- Handle missing values, outliers, and stationarity
- Cache fetched data to avoid redundant API calls
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from cpi_nowcast.config import (
    ADF_SIGNIFICANCE,
    CACHE_DIR,
    HISTORY_START,
    WINSOR_LOWER,
    WINSOR_UPPER,
)

logger = logging.getLogger(__name__)

# Cache file paths
CACHE_RAW = CACHE_DIR / "raw_data.parquet"
CACHE_PROCESSED = CACHE_DIR / "processed_data.parquet"


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet with metadata."""
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.info("Cached %d rows × %d cols to %s", len(df), df.shape[1], path)


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Load DataFrame from parquet cache; returns None if file missing."""
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded cached data from %s (%d rows)", path, len(df))
    return df


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

def compute_yoy(series: pd.Series, freq: str = "MS") -> pd.Series:
    """
    Compute year-over-year growth rate for a series.

    For monthly data: (x_t / x_{t-12}) - 1
    For daily data the caller should aggregate to monthly first.

    Parameters
    ----------
    series : pd.Series
        Level series with DatetimeIndex.
    freq : str
        Expected frequency ('MS' monthly, 'D' daily).

    Returns
    -------
    pd.Series
        YoY growth rate (as decimal, e.g. 0.05 = 5 %).
    """
    if freq == "MS":
        lag = 12
    elif freq == "D":
        lag = 252
    elif freq == "W" or freq == "W-MON":
        lag = 52
    else:
        lag = 12  # default to monthly

    yoy = series / series.shift(lag) - 1
    yoy.name = series.name
    return yoy


def winsorise(series: pd.Series, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER) -> pd.Series:
    """Winsorise series at given percentile bounds."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def check_stationarity(series: pd.Series, significance: float = ADF_SIGNIFICANCE) -> bool:
    """
    Run ADF test.  Returns True if the series is stationary.

    Parameters
    ----------
    series : pd.Series
        Series to test (NaNs are dropped).
    significance : float
        p-value threshold.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        clean = series.dropna()
        if len(clean) < 10:
            return True  # too few observations, skip
        result = adfuller(clean, autolag="AIC")
        pvalue = result[1]
        is_stationary = pvalue < significance
        logger.debug(
            "ADF(%s): p=%.3f → %s",
            series.name, pvalue, "stationary" if is_stationary else "non-stationary",
        )
        return is_stationary
    except Exception as exc:
        logger.warning("ADF test failed for %s: %s", series.name, exc)
        return True


def ensure_stationarity(series: pd.Series) -> pd.Series:
    """
    Differentiates the series once if the ADF test rejects stationarity.

    Returns
    -------
    pd.Series
        Stationary series.
    """
    if not check_stationarity(series):
        logger.info("Differencing %s to achieve stationarity", series.name)
        series = series.diff()
    return series


def interpolate_missing(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """
    Fill short gaps (<= max_gap) via linear interpolation;
    use last-observation-carried-forward (LOCF) for longer gaps.
    """
    # Mark gap lengths
    is_na = series.isna()
    filled = series.copy()

    # Linear interpolation for small gaps
    filled = filled.interpolate(method="linear", limit=max_gap)
    # LOCF for remaining
    filled = filled.ffill()
    return filled


def daily_to_monthly(series: pd.Series, method: str = "mean") -> pd.Series:
    """
    Resample a daily series to monthly frequency.

    Parameters
    ----------
    method : str
        'mean' (default) or 'last'.
    """
    if method == "mean":
        return series.resample("MS").mean()
    return series.resample("MS").last()


# ---------------------------------------------------------------------------
# Main fetch + merge pipeline
# ---------------------------------------------------------------------------

def fetch_all_raw(
    start_date: str = HISTORY_START,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch all data sources and return a merged daily/monthly raw DataFrame.

    Each fetcher is called with error handling so that a failing source does
    not break the whole pipeline.

    Parameters
    ----------
    start_date : str
        Earliest date to fetch.
    use_cache : bool
        If True, return cached data when available.
    force_refresh : bool
        If True, re-fetch even when cached data exists.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with all raw series; daily index.
    """
    if use_cache and not force_refresh:
        cached = _load_parquet(CACHE_RAW)
        if cached is not None:
            return cached

    frames: list[pd.Series] = []

    # ── FRED ──────────────────────────────────────────────────────────────
    try:
        from cpi_nowcast.data.fetchers.fred import fetch_brent_oil, fetch_natural_gas
        oil = fetch_brent_oil(start_date=start_date)
        if not oil.empty:
            frames.append(oil)
        gas = fetch_natural_gas(start_date=start_date)
        if not gas.empty:
            frames.append(gas)
    except Exception as exc:
        logger.error("FRED fetch failed: %s", exc)

    # ── ECB ───────────────────────────────────────────────────────────────
    try:
        from cpi_nowcast.data.fetchers.ecb import fetch_usd_dkk, fetch_eur_dkk, fetch_ois_rates
        usd_dkk = fetch_usd_dkk(start_date=start_date)
        if not usd_dkk.empty:
            frames.append(usd_dkk)
        eur_dkk = fetch_eur_dkk(start_date=start_date)
        if not eur_dkk.empty:
            frames.append(eur_dkk)
        ois = fetch_ois_rates(start_date=start_date)
        if not ois.empty:
            frames.extend([ois[c] for c in ois.columns])
    except Exception as exc:
        logger.error("ECB fetch failed: %s", exc)

    # ── DST ───────────────────────────────────────────────────────────────
    try:
        from cpi_nowcast.data.fetchers.dst import fetch_cpi, fetch_consumer_confidence
        cpi = fetch_cpi(start_date=start_date)
        if not cpi.empty:
            frames.append(cpi)
        conf = fetch_consumer_confidence(start_date=start_date)
        if not conf.empty:
            frames.append(conf)
    except Exception as exc:
        logger.error("DST fetch failed: %s", exc)

    # ── Eurostat ──────────────────────────────────────────────────────────
    try:
        from cpi_nowcast.data.fetchers.eurostat import fetch_hicp_flash, fetch_dk_hicp
        hicp_ea = fetch_hicp_flash(country="EA", start_date=start_date)
        if not hicp_ea.empty:
            frames.append(hicp_ea)
        hicp_dk = fetch_dk_hicp(start_date=start_date)
        if not hicp_dk.empty:
            frames.append(hicp_dk)
    except Exception as exc:
        logger.error("Eurostat fetch failed: %s", exc)

    # ── Nord Pool ─────────────────────────────────────────────────────────
    try:
        from cpi_nowcast.data.fetchers.nordpool import fetch_elspot_prices
        elspot = fetch_elspot_prices(start_date=start_date)
        if not elspot.empty:
            frames.extend([elspot[c] for c in elspot.columns])
    except Exception as exc:
        logger.error("Nord Pool fetch failed: %s", exc)

    # ── Freight / BDI ─────────────────────────────────────────────────────
    try:
        from cpi_nowcast.data.fetchers.freight import fetch_bdi, fetch_fao_food_price_index
        bdi = fetch_bdi(start_date=start_date)
        if not bdi.empty:
            frames.append(bdi)
        fao = fetch_fao_food_price_index(start_date=start_date)
        if not fao.empty:
            frames.append(fao)
    except Exception as exc:
        logger.error("Freight/FAO fetch failed: %s", exc)

    # ── Google Trends ─────────────────────────────────────────────────────
    try:
        from cpi_nowcast.data.fetchers.gtrends import fetch_trends_monthly
        gt = fetch_trends_monthly(start_date=start_date)
        if not gt.empty:
            frames.extend([gt[c] for c in gt.columns])
    except Exception as exc:
        logger.error("Google Trends fetch failed: %s", exc)

    if not frames:
        logger.error("All data fetchers failed — pipeline has no data")
        return pd.DataFrame()

    raw = pd.concat(frames, axis=1)
    raw = raw.sort_index()

    if use_cache:
        _save_parquet(raw, CACHE_RAW)

    return raw


def build_monthly_dataset(
    raw: pd.DataFrame,
    target_col: str = "cpi",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a clean, monthly, stationary feature matrix and target series.

    Steps:
    1. Resample all daily/higher-frequency series to monthly means
    2. Compute YoY growth rates for all series
    3. Winsorise at 1st/99th percentiles
    4. Fill missing values (linear interpolation + LOCF)
    5. Verify stationarity; difference further if needed
    6. Return (X, y) where X is features and y is CPI yoy

    Parameters
    ----------
    raw : pd.DataFrame
        Wide raw DataFrame from fetch_all_raw().
    target_col : str
        Column name of the target variable.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X features, y target) — both monthly DatetimeIndex.
    """
    # Step 1: Resample to monthly
    monthly = raw.resample("MS").mean()

    # Step 2: YoY growth rates
    yoy = monthly.apply(compute_yoy)

    # Step 3: Winsorise
    yoy = yoy.apply(winsorise)

    # Step 4: Fill missing values
    yoy = yoy.apply(interpolate_missing)

    # Step 5: Check stationarity and difference if needed
    for col in yoy.columns:
        if not check_stationarity(yoy[col]):
            yoy[col] = yoy[col].diff()

    # Separate target and features
    if target_col not in yoy.columns:
        logger.warning(
            "Target column '%s' not found. Available: %s",
            target_col, yoy.columns.tolist()
        )
        y = pd.Series(name=target_col, dtype=float)
        X = yoy
    else:
        y = yoy[target_col].rename("cpi_yoy")
        X = yoy.drop(columns=[target_col, "hicp_dk"], errors="ignore")

    # Drop rows where target is NaN
    valid = y.dropna().index
    X = X.loc[X.index.isin(valid)]
    y = y.loc[valid]

    logger.info(
        "Monthly dataset: %d obs × %d features. Target: %s",
        len(y), X.shape[1], y.name,
    )
    return X, y


def load_processed(
    start_date: str = HISTORY_START,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed dataset from cache or rebuild from scratch.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X features, y target).
    """
    if not force_refresh:
        cached = _load_parquet(CACHE_PROCESSED)
        if cached is not None:
            y = cached["cpi_yoy"]
            X = cached.drop(columns=["cpi_yoy"])
            return X, y

    raw = fetch_all_raw(start_date=start_date, force_refresh=force_refresh)
    X, y = build_monthly_dataset(raw)

    # Save
    processed = pd.concat([y, X], axis=1)
    _save_parquet(processed, CACHE_PROCESSED)
    return X, y
