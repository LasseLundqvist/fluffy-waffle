"""
Expanding-window out-of-sample backtest for UK CPI nowcasting.

Mimics pseudo-real-time forecasting:
- Training window expands each period
- Minimum MIN_TRAIN_MONTHS before first forecast
- Benchmarks: random walk + AR(1)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from uk_cpi_nowcast.config import BACKTEST_START, MIN_TRAIN_MONTHS, MODEL_CFG
from uk_cpi_nowcast.evaluation.metrics import evaluate_all, rmse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark models
# ---------------------------------------------------------------------------

def random_walk_forecast(y: pd.Series) -> pd.Series:
    """Random walk: CPI_yoy(t) = CPI_yoy(t-1)."""
    return y.shift(1).rename("rw_forecast")


def ar1_forecast(y_train: pd.Series, horizon: int = 1) -> float:
    """Fit AR(1) on training data and return h-step-ahead point forecast."""
    try:
        model = AutoReg(y_train.dropna(), lags=1, old_names=False)
        res = model.fit()
        forecast = res.forecast(steps=horizon)
        return float(forecast.iloc[-1])
    except Exception as exc:
        logger.warning("AR(1) forecast failed: %s — using last value", exc)
        return float(y_train.dropna().iloc[-1])


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    bridge_kwargs: Optional[dict] = None,
    midas_kwargs: Optional[dict] = None,
    start_date: str = BACKTEST_START,
    min_train: int = MIN_TRAIN_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run expanding-window backtest for Bridge and MIDAS models.

    Parameters
    ----------
    X : pd.DataFrame
        Monthly feature matrix (YoY, stationary).
    y : pd.Series
        Monthly UK CPI YoY target.
    bridge_kwargs : dict, optional
        Keyword arguments for BridgeEquationModel.
    midas_kwargs : dict, optional
        Keyword arguments for MIDASModel.
    start_date : str
        First forecast origin.
    min_train : int
        Minimum months of training data before first forecast.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - forecasts_df: columns [actual, bridge, midas, ensemble, rw, ar1]
        - metrics_df: one row per model with evaluation metrics
    """
    from uk_cpi_nowcast.models.bridge import BridgeEquationModel
    from uk_cpi_nowcast.models.midas import MIDASModel
    from uk_cpi_nowcast.models.ensemble import EnsembleModel

    bridge_kwargs = bridge_kwargs or {}
    midas_kwargs = midas_kwargs or {}

    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]

    start_ts = pd.Timestamp(start_date)
    forecast_dates = y.index[y.index >= start_ts]

    records: List[dict] = []

    for t in forecast_dates:
        train_mask = y.index < t
        n_train = train_mask.sum()

        if n_train < min_train:
            continue

        y_train = y.loc[train_mask]
        X_train = X.loc[train_mask]
        y_actual = y.loc[t] if t in y.index else np.nan

        record: dict = {"date": t, "actual": y_actual}

        # ── Random Walk ─────────────────────────────────────────
        record["rw"] = float(y_train.iloc[-1]) if len(y_train) > 0 else np.nan

        # ── AR(1) ───────────────────────────────────────────────
        record["ar1"] = ar1_forecast(y_train)

        # ── Bridge Equation ─────────────────────────────────────
        try:
            bridge = BridgeEquationModel(**bridge_kwargs)
            bridge.fit(X_train, y_train)
            # Pass full X up to and including t so lag shifts have prior-row
            # context (a single-row DataFrame shifted by lag >= 1 gives NaN).
            X_context = X.loc[:t] if t in X.index else pd.DataFrame()
            if not X_context.empty:
                pred_all = bridge.predict(X_context)
                valid = pred_all.dropna()
                record["bridge"] = float(valid.iloc[-1]) if not valid.empty else np.nan
            else:
                record["bridge"] = np.nan
        except Exception as exc:
            logger.warning("Bridge forecast failed at %s: %s", t, exc)
            record["bridge"] = np.nan

        # ── MIDAS ───────────────────────────────────────────────
        try:
            midas = MIDASModel(**midas_kwargs)
            midas.fit(X_train, y_train)
            X_pred = X.loc[[t]] if t in X.index else pd.DataFrame()
            if not X_pred.empty:
                pred = midas.predict(X_pred)
                record["midas"] = float(pred.iloc[0]) if not pred.isna().all() else np.nan
            else:
                record["midas"] = np.nan
        except Exception as exc:
            logger.warning("MIDAS forecast failed at %s: %s", t, exc)
            record["midas"] = np.nan

        records.append(record)

    if not records:
        logger.error("Backtest produced no records.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")

    # ── Ensemble ─────────────────────────────────────────────────
    constituent_cols = [c for c in ["bridge", "midas"] if c in df.columns]
    if len(constituent_cols) >= 1:
        ensemble = EnsembleModel(method=MODEL_CFG.ensemble_method)
        ensemble.fit(
            df[constituent_cols].dropna(),
            df["actual"].dropna(),
        )
        df["ensemble"] = ensemble.predict(df[constituent_cols])

    # ── Compute metrics ──────────────────────────────────────────
    actual = df["actual"].dropna()
    rw_bench = df["rw"]
    ar1_bench = df["ar1"]

    metric_rows: List[dict] = []
    for col in ["bridge", "midas", "ensemble", "rw", "ar1"]:
        if col not in df.columns:
            continue
        metrics = evaluate_all(
            actual,
            df[col],
            benchmark_rw=rw_bench,
            benchmark_ar=ar1_bench,
            label=col,
        )
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows).set_index("model")

    logger.info(
        "Backtest complete: %d forecast periods\n%s",
        len(df), metrics_df[["rmse", "mae", "mfe"]].round(4).to_string(),
    )
    return df, metrics_df
