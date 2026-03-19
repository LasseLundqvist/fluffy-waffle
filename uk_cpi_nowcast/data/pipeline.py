"""
Data pipeline: orchestrates all fetchers, preprocesses data, caches to parquet.

Data sources for UK CPI nowcasting:
- FRED:     Brent oil (daily), Natural gas TTF (monthly), USD/GBP (daily),
            Fertilizer Price Index (monthly)
- ECB:      EUR/GBP exchange rate (daily), EUR OIS 2Y/5Y (daily)
- Eurostat: UK HICP (monthly) — primary target
- ONS:      UK CPI (monthly) — fallback target / feature
- Stooq:    Baltic Dry Index (monthly)
- FAO:      Food Price Index (monthly)
- pytrends: Google Trends PCA components (weekly → monthly, geo=GB)
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from uk_cpi_nowcast.config import (
    ADF_SIGNIFICANCE,
    CACHE_DIR,
    HISTORY_START,
    WINSOR_LOWER,
    WINSOR_UPPER,
)

logger = logging.getLogger(__name__)

CACHE_RAW = CACHE_DIR / "raw_data.parquet"
CACHE_PROCESSED = CACHE_DIR / "processed_data.parquet"


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.info("Cached %d rows × %d cols to %s", len(df), df.shape[1], path)


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
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
    """
    if freq == "MS":
        lag = 12
    elif freq == "D":
        lag = 252
    elif freq in ("W", "W-MON"):
        lag = 52
    else:
        lag = 12

    yoy = series / series.shift(lag) - 1
    yoy.name = series.name
    return yoy


def compute_mom(series: pd.Series) -> pd.Series:
    """
    Compute month-over-month growth rate: (x_t / x_{t-1}) - 1.

    MoM is less autocorrelated than YoY and allows AR-augmented models to
    beat the random walk benchmark more easily.
    """
    mom = series / series.shift(1) - 1
    mom.name = series.name
    return mom


def winsorise(
    series: pd.Series,
    lower: float = WINSOR_LOWER,
    upper: float = WINSOR_UPPER,
) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def check_stationarity(series: pd.Series, significance: float = ADF_SIGNIFICANCE) -> bool:
    """Run ADF test. Returns True if the series is stationary."""
    try:
        from statsmodels.tsa.stattools import adfuller
        clean = series.dropna()
        if len(clean) < 10:
            return True
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


def interpolate_missing(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """Linear interpolation for short gaps; LOCF for longer gaps."""
    filled = series.copy()
    filled = filled.interpolate(method="linear", limit=max_gap)
    filled = filled.ffill()
    return filled


def daily_to_monthly(series: pd.Series, method: str = "mean") -> pd.Series:
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
    Fetch all data sources and return a merged raw DataFrame.

    Each fetcher is called with error handling so that a failing source does
    not break the whole pipeline.

    Data sources and their roles:
    - brent_oil:      Global energy cost → UK fuel & transport CPI (lag ~1-2m)
    - natural_gas:    UK gas bills & industrial energy costs (lag ~0-3m)
    - usd_gbp:        Import price channel (oil, commodities priced in USD)
    - eur_gbp:        European trade channel (40% of UK imports from EU)
    - ois_2y/5y:      Rate expectations → mortgage costs, credit conditions
    - fertilizer:     Agricultural input costs → food CPI (lag ~3-9m)
    - bdi:            Global shipping costs → import prices (lag ~1-3m)
    - fao_food_price: Global food commodity prices (lag ~1-6m)
    - hicp_uk:        UK HICP — primary target (Eurostat)
    - cpi_uk:         UK CPI — ONS official series (fallback / feature)
    - hicp_flash_ea:  EA HICP flash — correlated via trade (same-month indicator)
    - GT_PC*:         Google Trends PCA — consumer inflation expectations (nowcast)

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with all raw series.
    """
    if use_cache and not force_refresh:
        cached = _load_parquet(CACHE_RAW)
        if cached is not None:
            return cached

    frames: list[pd.Series] = []

    # ── FRED ──────────────────────────────────────────────────────────────
    try:
        from uk_cpi_nowcast.data.fetchers.fred import (
            fetch_brent_oil, fetch_natural_gas, fetch_usd_gbp, fetch_fertilizer_prices
        )
        for fn, desc in [
            (fetch_brent_oil, "Brent oil"),
            (fetch_natural_gas, "Natural gas TTF"),
            (fetch_usd_gbp, "USD/GBP"),
            (fetch_fertilizer_prices, "Fertilizer prices"),
        ]:
            try:
                s = fn(start_date=start_date)
                if not s.empty:
                    frames.append(s)
                    logger.info("Fetched: %s (%d obs)", desc, len(s))
            except Exception as exc:
                logger.warning("FRED %s fetch failed: %s", desc, exc)
    except Exception as exc:
        logger.error("FRED module import failed: %s", exc)

    # ── ECB ───────────────────────────────────────────────────────────────
    try:
        from uk_cpi_nowcast.data.fetchers.ecb import (
            fetch_eur_gbp, fetch_ois_rates, fetch_usd_gbp_cross
        )
        eur_gbp = fetch_eur_gbp(start_date=start_date)
        if not eur_gbp.empty:
            frames.append(eur_gbp)
        # USD/GBP cross rate (no FRED key needed)
        usd_gbp_cross = fetch_usd_gbp_cross(start_date=start_date)
        if not usd_gbp_cross.empty:
            frames.append(usd_gbp_cross)
        ois = fetch_ois_rates(start_date=start_date)
        if not ois.empty:
            frames.extend([ois[c] for c in ois.columns])
    except Exception as exc:
        logger.error("ECB fetch failed: %s", exc)

    # ── Eurostat UK HICP (primary target) ─────────────────────────────────
    try:
        from uk_cpi_nowcast.data.fetchers.eurostat import fetch_uk_hicp, fetch_ea_hicp_flash
        hicp_uk = fetch_uk_hicp(start_date=start_date)
        if not hicp_uk.empty:
            frames.append(hicp_uk)
        else:
            logger.warning("Eurostat UK HICP empty — will rely on ONS")
        hicp_ea = fetch_ea_hicp_flash(start_date=start_date)
        if not hicp_ea.empty:
            frames.append(hicp_ea)
    except Exception as exc:
        logger.error("Eurostat fetch failed: %s", exc)

    # ── ONS (fallback target and supplementary feature) ────────────────────
    try:
        from uk_cpi_nowcast.data.fetchers.ons import fetch_uk_cpi
        cpi_uk = fetch_uk_cpi(start_date=start_date)
        if not cpi_uk.empty:
            frames.append(cpi_uk)
    except Exception as exc:
        logger.error("ONS fetch failed: %s", exc)

    # ── Freight / BDI ─────────────────────────────────────────────────────
    try:
        from uk_cpi_nowcast.data.fetchers.freight import fetch_bdi, fetch_fao_food_price_index
        bdi = fetch_bdi(start_date=start_date)
        if not bdi.empty:
            frames.append(bdi)
        fao = fetch_fao_food_price_index(start_date=start_date)
        if not fao.empty:
            frames.append(fao)
    except Exception as exc:
        logger.error("Freight/FAO fetch failed: %s", exc)

    # ── Google Trends (geo=GB) ─────────────────────────────────────────────
    try:
        from uk_cpi_nowcast.data.fetchers.gtrends import fetch_trends_monthly
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


# ---------------------------------------------------------------------------
# Monthly dataset builder
# ---------------------------------------------------------------------------

def _resolve_target(monthly: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Determine the best available CPI target column.

    Priority:
    1. hicp_uk  (Eurostat UK HICP — preferred for consistency with DK pipeline)
    2. cpi_uk   (ONS CPI All Items — authoritative UK source)

    If the primary has gaps in recent periods, the secondary is used to fill them.
    Returns the merged target series name and the updated DataFrame.
    """
    hicp_col = "hicp_uk"
    ons_col = "cpi_uk"
    target_col = "cpi_target"

    if hicp_col in monthly.columns and not monthly[hicp_col].isna().all():
        s = monthly[hicp_col].copy()
        # Fill recent gaps from ONS if available
        if ons_col in monthly.columns:
            gap_mask = s.isna()
            if gap_mask.any():
                # Scale ONS to match Eurostat base (both should be 2015=100)
                # Simple level-matching using overlapping period
                overlap = monthly.loc[~gap_mask & monthly[ons_col].notna()]
                if not overlap.empty:
                    scale = overlap[hicp_col].mean() / overlap[ons_col].mean()
                    s.loc[gap_mask] = monthly.loc[gap_mask, ons_col] * scale
                    n_filled = gap_mask.sum()
                    logger.info("Target: filled %d gaps in hicp_uk using scaled cpi_uk", n_filled)
        monthly[target_col] = s
        return target_col, monthly

    if ons_col in monthly.columns and not monthly[ons_col].isna().all():
        logger.info("Target: using ONS cpi_uk as primary (Eurostat hicp_uk unavailable)")
        monthly[target_col] = monthly[ons_col].copy()
        return target_col, monthly

    logger.error("No UK CPI target series available in the data")
    return target_col, monthly


def build_monthly_dataset(
    raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a clean, monthly, stationary feature matrix and target series.

    Steps:
    1. Resample all series to monthly means
    2. Resolve the best CPI target (Eurostat HICP or ONS fallback)
    3. Compute YoY growth rates
    4. Process TARGET separately: YoY → dropna (no ffill, no differencing)
    5. Process FEATURES: YoY → winsorise → interpolate/ffill → stationarity diff
    6. Align X rows with known y; keep the latest X row for nowcasting even if
       the corresponding y is not yet available

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X: monthly feature matrix (includes latest row for nowcasting).
        y: CPI YoY only for periods where the outcome is known (no NaN).
    """
    # Step 1: Resample to monthly
    monthly = raw.resample("MS").mean()

    # Step 2: Resolve target
    target_col, monthly = _resolve_target(monthly)

    # Step 3a: Target — MoM, then keep only genuinely observed values.
    # MoM is less autocorrelated than YoY: the random walk is a weaker baseline
    # and AR-augmented bridge models can beat it.
    drop_cols = [target_col, "hicp_uk", "cpi_uk", "cpih_uk"]
    if target_col in monthly.columns:
        target_raw = monthly[target_col].copy()
        target_mom = compute_mom(target_raw)
        y = target_mom.dropna().rename("cpi_mom")
    else:
        logger.error("No UK CPI target series available in the data")
        y = pd.Series(name="cpi_mom", dtype=float)

    # Step 3b: Features — drop target-related columns to avoid leakage
    feat_cols = [c for c in monthly.columns if c not in drop_cols]
    feat_monthly = monthly[feat_cols]

    # YoY growth rates for all feature columns (commodity/FX in YoY captures
    # persistent supply-side effects; MoM target + YoY features is standard)
    feat_yoy = feat_monthly.apply(compute_yoy)

    # Step 4: Winsorise features
    feat_yoy = feat_yoy.apply(winsorise)

    # Step 5: Fill missing values (interpolation + LOCF) — features only
    feat_yoy = feat_yoy.apply(interpolate_missing)

    # Step 6: Check stationarity and difference features if needed
    for col in feat_yoy.columns:
        if not check_stationarity(feat_yoy[col]):
            feat_yoy[col] = feat_yoy[col].diff()

    # Step 7: Add AR features — lagged CPI MoM (legitimately known at forecast
    # time: lag-1 is last month's published figure).
    # Lag-12 captures the same-month seasonal baseline.
    for lag in [1, 2, 12]:
        ar_col = f"cpi_mom_lag{lag}"
        feat_yoy[ar_col] = y.shift(lag).reindex(feat_yoy.index)

    # X keeps all rows (including the latest where y may not yet be available)
    # Training alignment is handled inside fit() via dropna on (X_lagged, y)
    X = feat_yoy.dropna(how="all")

    logger.info(
        "Monthly dataset: %d target obs (MoM), %d feature rows x %d features (incl. AR lags)",
        len(y), len(X), X.shape[1],
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
            target_col = "cpi_mom" if "cpi_mom" in cached.columns else "cpi_yoy"
            y = cached[target_col].dropna()
            X = cached.drop(columns=[target_col])
            return X, y

    # force_refresh=True rebuilds the processed cache from raw.
    # Raw data is NOT re-fetched here; call fetch_all_raw(force_refresh=True)
    # before this if a full data refresh is needed (as --update-data does).
    raw = fetch_all_raw(start_date=start_date, force_refresh=False)
    X, y = build_monthly_dataset(raw)

    processed = pd.concat([y, X], axis=1)
    _save_parquet(processed, CACHE_PROCESSED)
    return X, y
