"""
MIDAS (Mixed Data Sampling) regression model.

Uses Almon polynomial lag weights to combine high-frequency predictors
with a monthly target variable.  This implementation works with
pre-aggregated monthly data (same as the bridge equation), but with
polynomial-constrained lag weights rather than freely estimated lags.

For a true MIDAS with daily data, users can pass daily series stacked
as multiple lag columns — the Almon weight function constrains those lags
to a smooth polynomial shape.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from cpi_nowcast.config import MODEL_CFG

logger = logging.getLogger(__name__)


def almon_weights(theta: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Compute Almon polynomial lag weights.

    w(k) = exp(θ₁·k + θ₂·k²) / Σⱼ exp(θ₁·j + θ₂·j²)

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector for the polynomial (length = degree).
    n_lags : int
        Number of lags.

    Returns
    -------
    np.ndarray
        Normalised weights of shape (n_lags,).
    """
    k = np.arange(1, n_lags + 1, dtype=float)
    degree = len(theta)
    exponent = sum(theta[d] * k ** (d + 1) for d in range(degree))
    # Numerical stability
    exponent -= exponent.max()
    w = np.exp(exponent)
    return w / w.sum()


def _build_midas_matrix(
    X_daily: pd.DataFrame,
    n_lags: int,
    freq: str = "MS",
) -> np.ndarray:
    """
    Stack lagged daily/weekly observations for each monthly target period.

    For each monthly period t, collects the last n_lags observations of each
    daily predictor column.

    Parameters
    ----------
    X_daily : pd.DataFrame
        High-frequency predictor DataFrame (daily or weekly, DatetimeIndex).
    n_lags : int
        Number of high-frequency lags per period.
    freq : str
        Target frequency (monthly = 'MS').

    Returns
    -------
    np.ndarray
        Shape (n_monthly, n_predictors * n_lags).
    """
    monthly_dates = pd.date_range(
        X_daily.index.min(), X_daily.index.max(), freq=freq
    )
    n_periods = len(monthly_dates)
    n_predictors = X_daily.shape[1]
    result = np.full((n_periods, n_predictors * n_lags), np.nan)

    for i, month_start in enumerate(monthly_dates):
        # Observations available up to (but not including) month_start
        mask = X_daily.index < month_start
        available = X_daily.loc[mask]
        if len(available) < n_lags:
            continue
        block = available.iloc[-n_lags:].values  # shape (n_lags, n_pred)
        result[i] = block.T.flatten()  # interleave: [pred0_lag1, …, pred0_lagN, pred1_lag1, …]

    return result, monthly_dates


class MIDASModel:
    """
    MIDAS regression with Almon polynomial weights.

    When daily predictor data is not available, this model degrades
    gracefully to a standard linear regression with polynomial-weighted lags.

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags per monthly period.
    degree : int
        Degree of the Almon polynomial.
    lambda_reg : float
        Ridge-like L2 regularisation for the outer coefficients.
    """

    def __init__(
        self,
        n_lags: int = 22,           # ~1 month of trading days
        degree: int = MODEL_CFG.midas_degree,
        lambda_reg: float = MODEL_CFG.ridge_alpha,
    ) -> None:
        self.n_lags = n_lags
        self.degree = degree
        self.lambda_reg = lambda_reg

        self._scaler = StandardScaler()
        self._theta: dict[str, np.ndarray] = {}   # Almon params per predictor
        self._beta: Optional[np.ndarray] = None   # Outer regression coefficients
        self._intercept: float = 0.0
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_predictor(
        self, X_col: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Contract a (n_obs, n_lags) matrix to (n_obs,) using Almon weights.

        Parameters
        ----------
        X_col : np.ndarray
            Shape (n_obs, n_lags).
        theta : np.ndarray
            Almon parameters.
        """
        w = almon_weights(theta, self.n_lags)
        return X_col @ w   # (n_obs,)

    def _objective(
        self,
        params: np.ndarray,
        X_stacked: np.ndarray,
        y: np.ndarray,
        n_predictors: int,
    ) -> float:
        """
        Objective for joint optimisation of Almon params + outer coefficients.

        params layout: [theta_0 … theta_{degree-1}] repeated for each predictor,
                       then outer betas.
        """
        theta_len = self.degree * n_predictors
        theta_flat = params[:theta_len]
        beta = params[theta_len:]

        # Build weighted predictors
        Z = np.zeros((X_stacked.shape[0], n_predictors))
        for j in range(n_predictors):
            theta_j = theta_flat[j * self.degree:(j + 1) * self.degree]
            X_j = X_stacked[:, j * self.n_lags:(j + 1) * self.n_lags]
            Z[:, j] = self._weighted_predictor(X_j, theta_j)

        intercept = beta[0]
        coef = beta[1:]
        yhat = intercept + Z @ coef
        resid = y - yhat
        mse = np.mean(resid ** 2)
        reg = self.lambda_reg * np.sum(coef ** 2)
        return mse + reg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_with_daily(
        self, X_daily: pd.DataFrame, y: pd.Series
    ) -> "MIDASModel":
        """
        Fit MIDAS using high-frequency (daily/weekly) predictor data.

        Parameters
        ----------
        X_daily : pd.DataFrame
            High-frequency predictors with DatetimeIndex.
        y : pd.Series
            Monthly CPI YoY target.
        """
        self._feature_cols = X_daily.columns.tolist()
        n_predictors = len(self._feature_cols)

        X_stacked, monthly_dates = _build_midas_matrix(X_daily, self.n_lags)

        # Align with y
        df = pd.DataFrame(
            X_stacked, index=monthly_dates
        ).join(y.rename("y"), how="inner").dropna()
        if df.empty:
            raise ValueError("MIDAS: no aligned observations after joining X and y.")

        y_arr = df["y"].values
        X_arr = df.drop(columns=["y"]).values

        # Initialise parameters
        theta_init = np.zeros(self.degree * n_predictors)
        beta_init = np.zeros(n_predictors + 1)
        beta_init[0] = y_arr.mean()
        params_init = np.concatenate([theta_init, beta_init])

        result = minimize(
            self._objective,
            params_init,
            args=(X_arr, y_arr, n_predictors),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        if not result.success:
            logger.warning("MIDAS optimisation did not converge: %s", result.message)

        theta_len = self.degree * n_predictors
        theta_flat = result.x[:theta_len]
        beta = result.x[theta_len:]
        self._intercept = beta[0]
        self._beta = beta[1:]
        for j, col in enumerate(self._feature_cols):
            self._theta[col] = theta_flat[j * self.degree:(j + 1) * self.degree]

        logger.info(
            "MIDAS fitted on %d monthly obs, %d predictors, %d lags each",
            len(y_arr), n_predictors, self.n_lags,
        )
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIDASModel":
        """
        Fit MIDAS using monthly predictor data (fallback mode).

        In this mode, each monthly observation is treated as n_lags=1
        and the Almon polynomial degenerates to a standard weight of 1.
        The model is equivalent to Ridge regression.

        Parameters
        ----------
        X : pd.DataFrame
            Monthly predictor matrix (YoY growth rates).
        y : pd.Series
            Monthly CPI YoY target.
        """
        self._feature_cols = X.columns.tolist()
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        if df.empty:
            raise ValueError("MIDAS (monthly fallback): no valid observations.")

        y_arr = df["y"].values
        X_arr = df.drop(columns=["y"]).values

        X_scaled = self._scaler.fit_transform(X_arr)

        # Simple ridge as fallback
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=self.lambda_reg)
        reg.fit(X_scaled, y_arr)
        self._intercept = reg.intercept_
        self._beta = reg.coef_

        logger.info(
            "MIDAS (monthly fallback) fitted on %d obs, %d predictors",
            len(y_arr), X_arr.shape[1],
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate nowcast predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with same columns as training X.

        Returns
        -------
        pd.Series
            Predicted CPI YoY.
        """
        if self._beta is None:
            raise RuntimeError("Model is not fitted.")

        X_aligned = X.reindex(columns=self._feature_cols)
        mask = X_aligned.notna().all(axis=1)
        result = pd.Series(np.nan, index=X.index, name="midas_nowcast")
        if mask.sum() == 0:
            return result

        X_scaled = self._scaler.transform(X_aligned.loc[mask].fillna(0).values)
        preds = self._intercept + X_scaled @ self._beta
        result.loc[mask] = preds
        return result
