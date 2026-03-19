"""
Forecast evaluation metrics for UK CPI nowcasting.

Implements RMSE, MAE, MFE (bias), skill scores, and the Diebold-Mariano test.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def rmse(actuals: pd.Series, forecasts: pd.Series) -> float:
    """Root Mean Squared Error."""
    err = (actuals - forecasts).dropna()
    return float(np.sqrt(np.mean(err ** 2)))


def mae(actuals: pd.Series, forecasts: pd.Series) -> float:
    """Mean Absolute Error."""
    err = (actuals - forecasts).dropna()
    return float(np.mean(np.abs(err)))


def mfe(actuals: pd.Series, forecasts: pd.Series) -> float:
    """Mean Forecast Error (bias). Positive = over-forecast."""
    err = (forecasts - actuals).dropna()
    return float(np.mean(err))


def skill_score(
    actuals: pd.Series,
    forecasts: pd.Series,
    benchmark: pd.Series,
    metric: str = "rmse",
) -> float:
    """
    Skill score relative to a benchmark.

    skill = 1 - metric(model) / metric(benchmark)
    Positive = model beats benchmark.
    """
    fn = {"rmse": rmse, "mae": mae}.get(metric, rmse)
    m_model = fn(actuals, forecasts)
    m_bench = fn(actuals, benchmark)
    if m_bench == 0:
        return 0.0
    return 1.0 - m_model / m_bench


def diebold_mariano_test(
    actuals: pd.Series,
    forecasts_1: pd.Series,
    forecasts_2: pd.Series,
    loss: str = "squared",
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H₀: E[d_t] = 0  where d_t = L(e1_t) - L(e2_t)

    Returns
    -------
    (dm_stat, p_value) : Tuple[float, float]
    """
    idx = actuals.index.intersection(forecasts_1.index).intersection(forecasts_2.index)
    e1 = actuals.loc[idx] - forecasts_1.loc[idx]
    e2 = actuals.loc[idx] - forecasts_2.loc[idx]

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) - np.abs(e2)

    d = d.dropna()
    n = len(d)
    if n < 10:
        logger.warning("DM test: too few observations (%d)", n)
        return np.nan, np.nan

    d_mean = d.mean()
    gamma = [
        np.cov(d.values[:-j] if j > 0 else d.values,
                d.values[j:] if j > 0 else d.values)[0, 1]
        for j in range(h)
    ]
    var_d = gamma[0] / n + 2.0 * sum(gamma[1:]) / n
    if var_d <= 0:
        var_d = d.var() / n

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def evaluate_all(
    actuals: pd.Series,
    forecasts: pd.Series,
    benchmark_rw: Optional[pd.Series] = None,
    benchmark_ar: Optional[pd.Series] = None,
    label: str = "model",
) -> dict:
    """Compute all metrics for a single model."""
    results: dict = {
        "model": label,
        "n": int(actuals.dropna().shape[0]),
        "rmse": rmse(actuals, forecasts),
        "mae": mae(actuals, forecasts),
        "mfe": mfe(actuals, forecasts),
    }

    if benchmark_rw is not None:
        results["skill_vs_rw"] = skill_score(actuals, forecasts, benchmark_rw)
        dm_stat, dm_p = diebold_mariano_test(actuals, forecasts, benchmark_rw)
        results["dm_stat_vs_rw"] = dm_stat
        results["dm_pval_vs_rw"] = dm_p

    if benchmark_ar is not None:
        results["skill_vs_ar"] = skill_score(actuals, forecasts, benchmark_ar)
        dm_stat, dm_p = diebold_mariano_test(actuals, forecasts, benchmark_ar)
        results["dm_stat_vs_ar"] = dm_stat
        results["dm_pval_vs_ar"] = dm_p

    return results
