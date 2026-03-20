"""
UK CPI Nowcasting — Standalone Script
======================================
Structure:
  1. Configuration
  2. Data Extraction   (ONS, yfinance, ECB)
  3. Transformation    (MoM, AR features, winsorising, stationarity)
  4. Regressions       (Bridge + Ridge-MIDAS + ensemble, h=1..6)
  5. Output            (print + CSV)

Requirements (install once):
    pip install pandas numpy scipy scikit-learn requests yfinance statsmodels

Optional (adds oil/gas/FX from FRED instead of yfinance):
    pip install fredapi
    Set environment variable: FRED_API_KEY=your_key_here

Run:
    python uk_cpi_nowcast_standalone.py
"""

# =============================================================================
# 0. Imports
# =============================================================================
import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import mstats
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Configuration
# =============================================================================

HISTORY_START = "2010-01-01"   # Earliest data to fetch
BACKTEST_START = "2015-01-01"  # First backtest forecast origin
MIN_TRAIN_MONTHS = 36          # Minimum months before first backtest forecast
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")  # Optional

RIDGE_ALPHA = 1.0
MAX_LAGS = 3          # Lags 0..MAX_LAGS tested per variable in Bridge
N_CV_SPLITS = 5       # Time-series CV folds for lag selection
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

OUTPUT_DIR = Path(__file__).parent  # Save CSVs next to this script


# =============================================================================
# 2. Data Extraction
# =============================================================================

# ── 2a. ONS (no API key required) ────────────────────────────────────────────

def fetch_ons_cpi(start_date: str = HISTORY_START) -> pd.Series:
    """UK CPI All Items Index (2015=100) from ONS MM23 dataset. Series D7BT."""
    url = "https://api.ons.gov.uk/v1/data"
    uri = "/economy/inflationandpriceindices/timeseries/d7bt/mm23"
    try:
        resp = requests.get(url, params={"uri": uri}, timeout=30)
        resp.raise_for_status()
        months = resp.json().get("months", [])
        rows = []
        for item in months:
            try:
                dt = pd.to_datetime(item["date"], format="%Y %b")
                rows.append({"date": dt, "value": float(item["value"])})
            except (KeyError, ValueError):
                continue
        s = pd.DataFrame(rows).set_index("date")["value"].sort_index()
        s = s[s.index >= pd.Timestamp(start_date)]
        s.name = "cpi_uk"
        logger.info("ONS: fetched %d CPI observations", len(s))
        return s
    except Exception as exc:
        logger.error("ONS fetch failed: %s", exc)
        return pd.Series(name="cpi_uk", dtype=float)


# ── 2b. yfinance (no API key required) ───────────────────────────────────────

def _yfinance_monthly(ticker: str, name: str, start: str, invert: bool = False) -> pd.Series:
    """Download monthly close prices from Yahoo Finance."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, interval="1mo",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.Series(name=name, dtype=float)
        close = raw["Close"].squeeze()
        if invert:
            close = 1.0 / close
        close.index = pd.DatetimeIndex(
            [pd.Timestamp(d.year, d.month, 1) for d in pd.to_datetime(close.index)]
        )
        close = close[close.index >= pd.Timestamp(start)]
        close.name = name
        logger.info("yfinance: fetched %d obs for %s (%s)", len(close), name, ticker)
        return close
    except ImportError:
        logger.warning("yfinance not installed. Run: pip install yfinance")
        return pd.Series(name=name, dtype=float)
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", name, exc)
        return pd.Series(name=name, dtype=float)


def fetch_brent_oil(start: str = HISTORY_START) -> pd.Series:
    """Brent crude oil price (USD/barrel). yfinance BZ=F."""
    return _yfinance_monthly("BZ=F", "brent_oil", start)


def fetch_natural_gas(start: str = HISTORY_START) -> pd.Series:
    """Henry Hub natural gas (USD/MMBtu). yfinance NG=F."""
    return _yfinance_monthly("NG=F", "natural_gas", start)


def fetch_usd_gbp(start: str = HISTORY_START) -> pd.Series:
    """USD per GBP exchange rate. yfinance GBPUSD=X (inverted)."""
    return _yfinance_monthly("GBPUSD=X", "usd_gbp", start, invert=False)


def fetch_ftse100(start: str = HISTORY_START) -> pd.Series:
    """FTSE 100 index. yfinance ^FTSE."""
    return _yfinance_monthly("^FTSE", "ftse100", start)


# ── 2c. ECB (no API key required) ─────────────────────────────────────────────

def fetch_ecb_series(series_key: str, name: str, start: str = HISTORY_START) -> pd.Series:
    """Fetch a daily series from ECB SDMX 2.1 API, resample to monthly."""
    url = f"https://data-api.ecb.europa.eu/service/data/{series_key}"
    params = {
        "startPeriod": start[:7],
        "detail": "dataonly",
        "format": "csvdata",
    }
    try:
        resp = requests.get(url, params=params, timeout=30, headers={"Accept": "text/csv"})
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        # ECB CSV: columns include TIME_PERIOD and OBS_VALUE
        time_col = next((c for c in df.columns if "TIME" in c.upper()), None)
        val_col = next((c for c in df.columns if "OBS_VALUE" in c.upper()), None)
        if time_col is None or val_col is None:
            return pd.Series(name=name, dtype=float)
        s = pd.Series(df[val_col].values, index=pd.to_datetime(df[time_col]))
        s = pd.to_numeric(s, errors="coerce").dropna()
        s = s.resample("MS").mean()
        s = s[s.index >= pd.Timestamp(start)]
        s.name = name
        logger.info("ECB: fetched %d monthly obs for %s", len(s), name)
        return s
    except Exception as exc:
        logger.warning("ECB fetch failed for %s: %s", name, exc)
        return pd.Series(name=name, dtype=float)


def fetch_eur_gbp(start: str = HISTORY_START) -> pd.Series:
    """EUR/GBP exchange rate from ECB."""
    return fetch_ecb_series("EXR/D.GBP.EUR.SP00.A", "eur_gbp", start)


# ── 2d. FRED (optional — requires API key) ───────────────────────────────────

def fetch_fred(series_id: str, name: str, start: str = HISTORY_START) -> pd.Series:
    """Fetch from FRED REST API. Returns empty Series if no API key."""
    if not FRED_API_KEY:
        return pd.Series(name=name, dtype=float)
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "observation_start": start,
        "file_type": "json",
        "api_key": FRED_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        records = {o["date"]: float(o["value"]) for o in obs if o["value"] != "."}
        s = pd.Series(records, name=name)
        s.index = pd.to_datetime(s.index)
        logger.info("FRED: fetched %d obs for %s (%s)", len(s), name, series_id)
        return s.sort_index()
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(name=name, dtype=float)


# ── 2e. Master fetch ──────────────────────────────────────────────────────────

def fetch_all(start: str = HISTORY_START) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Fetch all data sources and return:
      - y_raw : raw CPI index (monthly)
      - X_raw : DataFrame of predictor series (various frequencies, aligned monthly)
    """
    logger.info("=== Data Extraction ===")

    cpi = fetch_ons_cpi(start)

    # Commodity prices
    brent    = fetch_brent_oil(start)
    gas      = fetch_natural_gas(start)

    # FX
    usd_gbp  = fetch_usd_gbp(start)
    eur_gbp  = fetch_eur_gbp(start)

    # Equity
    ftse     = fetch_ftse100(start)

    # Optional FRED series (requires API key)
    fertilizer = fetch_fred("PFERTILIZERINDEXM", "fertilizer", start)

    # Combine predictors into monthly DataFrame
    series_list = [brent, gas, usd_gbp, eur_gbp, ftse, fertilizer]
    monthly: Dict[str, pd.Series] = {}
    for s in series_list:
        if s.empty:
            continue
        # Resample daily series to monthly mean
        if s.index.freq is None and len(s) > 12 and (s.index[-1] - s.index[0]).days > 365:
            s_m = s.resample("MS").mean()
        else:
            s_m = s.copy()
            s_m.index = s_m.index.to_period("M").to_timestamp()
        monthly[s.name] = s_m

    X_raw = pd.DataFrame(monthly)
    X_raw.index = pd.DatetimeIndex(X_raw.index).to_period("M").to_timestamp()
    X_raw = X_raw.sort_index()

    return cpi, X_raw


# =============================================================================
# 3. Transformation
# =============================================================================

def compute_mom(series: pd.Series) -> pd.Series:
    """Month-on-month growth rate: (x_t / x_{t-1}) - 1."""
    return (series / series.shift(1) - 1).rename(series.name)


def winsorise(df: pd.DataFrame, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER) -> pd.DataFrame:
    """Clip each column to [lower, upper] quantile range."""
    result = df.copy()
    for col in result.columns:
        clean = result[col].dropna()
        if len(clean) < 10:
            continue
        lo, hi = np.quantile(clean, [lower, upper])
        result[col] = result[col].clip(lo, hi)
    return result


def make_stationary(series: pd.Series) -> pd.Series:
    """
    Apply YoY growth rate to make a level series stationary.
    Skips series that are already returns/rates (std < 0.5 and mean near 0).
    """
    clean = series.dropna()
    if len(clean) < 24:
        return series
    # Heuristic: if values look like levels (e.g., index > 50), apply YoY
    if clean.abs().mean() > 5:
        yoy = series / series.shift(12) - 1
        return yoy.rename(series.name)
    return series


def build_features(cpi_raw: pd.Series, X_raw: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Transform raw data into model-ready monthly features.

    Returns
    -------
    y : pd.Series
        CPI MoM (the target).
    X : pd.DataFrame
        Feature matrix — YoY growth rates + AR features, winsorised.
    """
    logger.info("=== Transformation ===")

    # Target: MoM
    cpi_monthly = cpi_raw.resample("MS").last().dropna()
    y = compute_mom(cpi_monthly).dropna()
    y.name = "cpi_mom"

    # Predictors: YoY growth rates
    feat: Dict[str, pd.Series] = {}
    for col in X_raw.columns:
        s = X_raw[col].dropna()
        if len(s) < 24:
            continue
        feat[col] = make_stationary(s)

    X = pd.DataFrame(feat)
    X.index = pd.DatetimeIndex(X.index).to_period("M").to_timestamp()
    X = X.sort_index()

    # Winsorise predictors
    X = winsorise(X)

    # AR features: lagged CPI MoM as autoregressive anchors
    for lag in [1, 2, 12]:
        X[f"cpi_mom_lag{lag}"] = y.shift(lag).reindex(X.index)

    # Align y and X to common monthly index
    common = y.index.intersection(X.index)
    y = y.loc[common]
    X = X.loc[common]

    logger.info(
        "Features built: %d monthly obs, %d predictors, target range %s–%s",
        len(y), X.shape[1],
        y.index[0].strftime("%Y-%m"), y.index[-1].strftime("%Y-%m"),
    )
    return y, X


# =============================================================================
# 4. Regressions
# =============================================================================

# ── 4a. Bridge Equation with lag selection ────────────────────────────────────

def select_lags(X: pd.DataFrame, y: pd.Series, max_lags: int = MAX_LAGS) -> Dict[str, int]:
    """
    For each predictor, pick lag 0..max_lags that minimises out-of-sample MSE
    in time-series cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    best_lags: Dict[str, int] = {}
    for col in X.columns:
        best_mse = np.inf
        best_lag = 0
        for lag in range(0, max_lags + 1):
            x_lag = X[[col]].shift(lag)
            df_cv = pd.concat([y, x_lag], axis=1).dropna()
            if len(df_cv) < 20:
                continue
            y_cv = df_cv.iloc[:, 0].values
            X_cv = df_cv.iloc[:, 1:].values
            fold_mses = []
            for tr, va in tscv.split(X_cv):
                if len(va) == 0:
                    continue
                reg = Ridge(alpha=RIDGE_ALPHA)
                reg.fit(X_cv[tr], y_cv[tr])
                fold_mses.append(np.mean((reg.predict(X_cv[va]) - y_cv[va]) ** 2))
            if fold_mses and np.mean(fold_mses) < best_mse:
                best_mse = np.mean(fold_mses)
                best_lag = lag
        best_lags[col] = best_lag
    return best_lags


def build_lagged_X(X: pd.DataFrame, lags: Dict[str, int]) -> pd.DataFrame:
    cols = []
    for col, lag in lags.items():
        if col not in X.columns:
            continue
        s = X[col].shift(lag)
        s.name = f"{col}_L{lag}"
        cols.append(s)
    return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=X.index)


class BridgeModel:
    """Ridge regression with automatic lag selection per predictor."""

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.lags_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self._scaler = StandardScaler()
        self._reg = Ridge(alpha=RIDGE_ALPHA)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BridgeModel":
        y_h = y.shift(-self.horizon) if self.horizon > 1 else y
        self.lags_ = select_lags(X, y_h)
        X_lag = build_lagged_X(X, self.lags_)
        X_lag = X_lag.dropna(axis=1, how="all")
        df = pd.concat([y_h, X_lag], axis=1).dropna()
        if df.empty:
            raise ValueError("No valid training rows.")
        y_tr = df.iloc[:, 0].values
        X_tr = df.iloc[:, 1:].values
        self.feature_names_ = df.columns[1:].tolist()
        self._scaler.fit(X_tr)
        self._reg.fit(self._scaler.transform(X_tr), y_tr)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_lag = build_lagged_X(X, self.lags_)
        X_lag = X_lag.reindex(columns=self.feature_names_)
        mask = X_lag.notna().all(axis=1)
        result = pd.Series(np.nan, index=X.index)
        if mask.sum() == 0:
            return result
        result.loc[mask] = self._reg.predict(
            self._scaler.transform(X_lag.loc[mask].values)
        )
        return result


# ── 4b. MIDAS (Ridge fallback on monthly data) ────────────────────────────────

class MIDASModel:
    """Ridge regression on monthly data (MIDAS fallback without high-freq data)."""

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.feature_cols_: List[str] = []
        self._scaler = StandardScaler()
        self._reg = Ridge(alpha=RIDGE_ALPHA)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIDASModel":
        y_h = y.shift(-self.horizon) if self.horizon > 1 else y
        X_clean = X.dropna(axis=1, how="all")
        self.feature_cols_ = X_clean.columns.tolist()
        df = pd.concat([y_h.rename("y"), X_clean], axis=1).dropna()
        if df.empty:
            raise ValueError("No valid training rows.")
        y_tr = df["y"].values
        X_tr = df.drop(columns=["y"]).values
        self._scaler.fit(X_tr)
        self._reg.fit(self._scaler.transform(X_tr), y_tr)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_al = X.reindex(columns=self.feature_cols_)
        mask = X_al.notna().all(axis=1)
        result = pd.Series(np.nan, index=X.index)
        if mask.sum() == 0:
            return result
        result.loc[mask] = self._reg.predict(
            self._scaler.transform(X_al.loc[mask].fillna(0).values)
        )
        return result


# ── 4c. Backtest ──────────────────────────────────────────────────────────────

def run_backtest(
    X: pd.DataFrame, y: pd.Series,
    start_date: str = BACKTEST_START,
    min_train: int = MIN_TRAIN_MONTHS,
) -> pd.DataFrame:
    """
    Expanding-window backtest for Bridge + MIDAS vs random walk benchmark.

    Returns DataFrame with columns: actual, bridge, midas, ensemble, rw.
    """
    logger.info("=== Backtest ===")
    idx = X.index.intersection(y.index)
    X, y = X.loc[idx], y.loc[idx]
    forecast_dates = y.index[y.index >= pd.Timestamp(start_date)]
    records = []

    for t in forecast_dates:
        train_mask = y.index < t
        if train_mask.sum() < min_train:
            continue
        y_tr, X_tr = y.loc[train_mask], X.loc[train_mask]
        y_actual = float(y.loc[t]) if t in y.index else np.nan
        rec = {"date": t, "actual": y_actual, "rw": float(y_tr.iloc[-1])}

        for name, Model in [("bridge", BridgeModel), ("midas", MIDASModel)]:
            try:
                m = Model(horizon=1)
                m.fit(X_tr, y_tr)
                if name == "bridge":
                    preds = m.predict(X.loc[:t])
                    valid = preds.dropna()
                    rec[name] = float(valid.iloc[-1]) if not valid.empty else np.nan
                else:
                    pred = m.predict(X.loc[[t]])
                    rec[name] = float(pred.iloc[0]) if not pred.isna().all() else np.nan
            except Exception as exc:
                logger.debug("%s failed at %s: %s", name, t, exc)
                rec[name] = np.nan

        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    df["ensemble"] = df[["bridge", "midas"]].mean(axis=1)

    # Metrics
    actual = df["actual"].dropna()
    logger.info("\nBacktest metrics (RMSE):")
    for col in ["bridge", "midas", "ensemble", "rw"]:
        aligned = df[col].reindex(actual.index).dropna()
        a = actual.reindex(aligned.index)
        rmse = np.sqrt(np.mean((a - aligned) ** 2))
        logger.info("  %-10s RMSE = %.4f", col, rmse)

    return df


# ── 4d. Nowcast + 6-month path ────────────────────────────────────────────────

def nowcast(
    X: pd.DataFrame, y: pd.Series,
    backtest_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fit on full history and produce nowcast + 6-month direct forecast path.

    Returns path_df: indexed by horizon (1..6), columns bridge/midas/ensemble.
    """
    logger.info("=== Nowcast ===")
    t = X.index[-1]
    path_records = []

    for h in range(1, 7):
        bridge = BridgeModel(horizon=h)
        midas = MIDASModel(horizon=h)
        try:
            bridge.fit(X, y)
            b_full = bridge.predict(X)
            b_val = float(b_full.dropna().iloc[-1]) if not b_full.dropna().empty else np.nan
        except Exception:
            b_val = np.nan
        try:
            midas.fit(X, y)
            m_full = midas.predict(X)
            m_val = float(m_full.dropna().iloc[-1]) if not m_full.dropna().empty else np.nan
        except Exception:
            m_val = np.nan

        ens_val = float(np.nanmean([b_val, m_val]))
        target_month = t + pd.DateOffset(months=h)
        path_records.append({
            "horizon": h,
            "target_month": target_month.strftime("%b %Y"),
            "bridge": b_val,
            "midas": m_val,
            "ensemble": ens_val,
        })

    path_df = pd.DataFrame(path_records).set_index("horizon")
    return path_df


# =============================================================================
# 5. Output
# =============================================================================

def print_results(y: pd.Series, path_df: pd.DataFrame) -> None:
    """Pretty-print the nowcast and forecast path."""
    t = y.index[-1] + pd.DateOffset(months=1)

    print(f"\n{'='*55}")
    print(f"  UK CPI Nowcasting — {datetime.now().strftime('%d %b %Y')}")
    print(f"{'='*55}")
    print(f"\n  Forecast origin: {(y.index[-1]).strftime('%B %Y')}")
    print(f"  Last observed CPI MoM: {y.iloc[-1]:+.3%}\n")

    print(f"  {'H':<4} {'Target':<12} {'Bridge':>9} {'MIDAS':>9} {'Ensemble':>10}")
    print(f"  {'-'*46}")
    for h, row in path_df.iterrows():
        b = f"{row['bridge']:+.3%}" if not np.isnan(row['bridge']) else "   N/A"
        m = f"{row['midas']:+.3%}"  if not np.isnan(row['midas'])  else "   N/A"
        e = f"{row['ensemble']:+.3%}" if not np.isnan(row['ensemble']) else "   N/A"
        print(f"  h={h:<2} {row['target_month']:<12} {b:>9} {m:>9} {e:>10}")

    cum = path_df["ensemble"].sum()
    last6 = y.iloc[-6:].sum()
    direction = "^ (rising)" if cum > last6 + 0.001 else "v (falling)" if cum < last6 - 0.001 else "-> (stable)"
    print(f"\n  Cumulative 6-month ensemble: {cum:+.3%}")
    print(f"  Last 6 months actual:        {last6:+.3%}")
    print(f"  Inflation trend:             {direction}")
    print(f"{'='*55}\n")


def save_outputs(
    path_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Save nowcast path and backtest results to CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # Forecast path
    path_out = OUTPUT_DIR / f"uk_cpi_forecast_path_{ts}.csv"
    path_df.to_csv(path_out)
    logger.info("Forecast path saved to: %s", path_out)

    # Backtest
    if not backtest_df.empty:
        bt_out = OUTPUT_DIR / f"uk_cpi_backtest_{ts}.csv"
        backtest_df.to_csv(bt_out)
        logger.info("Backtest saved to: %s", bt_out)

    # Actuals
    y_out = OUTPUT_DIR / f"uk_cpi_actuals_{ts}.csv"
    y.to_frame("cpi_mom").to_csv(y_out)
    logger.info("Actuals saved to: %s", y_out)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # 1. Extract
    cpi_raw, X_raw = fetch_all(HISTORY_START)
    if cpi_raw.empty:
        logger.error("CPI data unavailable. Check your internet connection.")
        sys.exit(1)

    # 2. Transform
    y, X = build_features(cpi_raw, X_raw)
    if y.empty or X.empty:
        logger.error("Feature building failed.")
        sys.exit(1)

    # 3. Backtest
    backtest_df = run_backtest(X, y)

    # 4. Nowcast
    path_df = nowcast(X, y, backtest_df)

    # 5. Output
    print_results(y, path_df)
    save_outputs(path_df, backtest_df, y)


if __name__ == "__main__":
    main()
