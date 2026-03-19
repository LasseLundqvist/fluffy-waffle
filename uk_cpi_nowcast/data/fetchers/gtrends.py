"""
Google Trends fetcher via pytrends — UK geography (geo=GB).

Applies PCA to reduce the keyword matrix to orthogonal components.
Respects Google Trends rate limits (max 5 requests/minute).

UK-specific keywords capture consumer awareness of inflation across
energy, food, housing, and general cost-of-living dimensions.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from uk_cpi_nowcast.config import (
    GTRENDS_GEO,
    GTRENDS_KEYWORDS,
    GTRENDS_PCA_VARIANCE,
    GTRENDS_RATE_LIMIT,
    GTRENDS_TIMEFRAME,
    HISTORY_START,
    MODEL_CFG,
)

logger = logging.getLogger(__name__)

_MIN_REQUEST_DELAY = 60.0 / GTRENDS_RATE_LIMIT


def _build_pytrends():
    """Import and return a TrendReq instance (lazy import)."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends is not installed. Install with: pip install pytrends")
        raise

    # pytrends 4.x uses urllib3 Retry(method_whitelist=...) which was renamed
    # to allowed_methods in urllib3 2.0.  Pass retries=0 to avoid the error.
    try:
        return TrendReq(hl="en-GB", tz=0, timeout=(10, 30), retries=0)
    except TypeError:
        # Older pytrends / urllib3 combo: fall back to no retry args
        return TrendReq(hl="en-GB", tz=0, timeout=(10, 30))


def fetch_trends_group(
    keywords: List[str],
    geo: str = GTRENDS_GEO,
    timeframe: str = GTRENDS_TIMEFRAME,
) -> pd.DataFrame:
    """
    Fetch Google Trends data for a list of keywords (UK geography).

    Parameters
    ----------
    keywords : list of str
        Search terms (max 5 per request).
    geo : str
        Country code, e.g. "GB".
    timeframe : str
        Pytrends timeframe string.

    Returns
    -------
    pd.DataFrame
        Weekly DatetimeIndex; columns per keyword (0–100 scale).
    """
    pt = _build_pytrends()
    frames: list[pd.DataFrame] = []

    batches = [keywords[i:i + 5] for i in range(0, len(keywords), 5)]
    for batch in batches:
        try:
            pt.build_payload(batch, cat=0, timeframe=timeframe, geo=geo, gprop="")
            df = pt.interest_over_time()
            if not df.empty:
                df = df.drop(columns=["isPartial"], errors="ignore")
                frames.append(df)
            time.sleep(_MIN_REQUEST_DELAY)
        except Exception as exc:
            logger.warning("Google Trends batch failed %s: %s", batch, exc)
            time.sleep(_MIN_REQUEST_DELAY * 2)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.sort_index()


def fetch_all_trends(
    keyword_groups: Dict[str, List[str]] = GTRENDS_KEYWORDS,
    geo: str = GTRENDS_GEO,
    timeframe: str = GTRENDS_TIMEFRAME,
) -> pd.DataFrame:
    """
    Fetch all configured Google Trends keyword groups.

    Returns
    -------
    pd.DataFrame
        Weekly DatetimeIndex; all keywords as columns.
    """
    frames: list[pd.DataFrame] = []
    for group_name, keywords in keyword_groups.items():
        logger.info("Fetching Google Trends group: %s (%s)", group_name, keywords)
        df = fetch_trends_group(keywords, geo=geo, timeframe=timeframe)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.sort_index()


def apply_pca_to_trends(
    df: pd.DataFrame,
    variance_threshold: float = GTRENDS_PCA_VARIANCE,
    random_state: int = MODEL_CFG.random_seed,
) -> Tuple[pd.DataFrame, object]:
    """
    Reduce Google Trends matrix via PCA.

    Keeps principal components that together explain at least
    ``variance_threshold`` of total variance (or those with eigenvalue > 1).

    Returns
    -------
    pd.DataFrame
        PC scores as columns GT_PC1, GT_PC2, …
    sklearn.decomposition.PCA
        Fitted PCA object.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df_clean = df.dropna().copy()
    if df_clean.empty:
        logger.warning("PCA: no clean rows in trends DataFrame")
        return pd.DataFrame(index=df.index), None

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean.values)

    pca_full = PCA(random_state=random_state)
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_variance = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_eigenvalue = int(np.sum(pca_full.explained_variance_ > 1))
    n_components = max(n_variance, n_eigenvalue, 1)
    n_components = min(n_components, X.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(X)
    cols = [f"GT_PC{i+1}" for i in range(n_components)]
    pc_df = pd.DataFrame(scores, index=df_clean.index, columns=cols)

    logger.info(
        "PCA on Google Trends: %d components explain %.1f%% variance",
        n_components,
        pca.explained_variance_ratio_.sum() * 100,
    )
    return pc_df, pca


def fetch_trends_monthly(
    start_date: str = HISTORY_START,
) -> pd.DataFrame:
    """
    Convenience: fetch UK trends, apply PCA, resample to monthly means.

    Returns
    -------
    pd.DataFrame
        Monthly DatetimeIndex; columns GT_PC1, GT_PC2, …
    """
    start_ts = pd.Timestamp(start_date)
    months_back = (pd.Timestamp.now() - start_ts).days // 30
    if months_back <= 12:
        tf = GTRENDS_TIMEFRAME
    elif months_back <= 36:
        tf = "today 3-y"
    elif months_back <= 60:
        tf = "today 5-y"
    else:
        tf = f"{start_ts.strftime('%Y-%m-%d')} {pd.Timestamp.now().strftime('%Y-%m-%d')}"

    try:
        weekly = fetch_all_trends(timeframe=tf)
        if weekly.empty:
            logger.warning("Google Trends (GB): empty result")
            return pd.DataFrame()

        pc_df, _ = apply_pca_to_trends(weekly)
        if pc_df.empty:
            return pd.DataFrame()

        monthly = pc_df.resample("MS").mean()
        return monthly[monthly.index >= pd.Timestamp(start_date)]

    except Exception as exc:
        logger.error("Google Trends fetch failed: %s", exc)
        return pd.DataFrame()
