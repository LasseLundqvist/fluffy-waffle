"""
Bridge Equation Model for CPI nowcasting.

Specification:
    CPI_yoy(t) = α + Σᵢ βᵢ·Xᵢ(t) + ε(t)

where all X are year-over-year growth rates of the high-frequency indicators,
temporally aggregated to the monthly frequency.

Regularisation: Ridge or ElasticNet (sklearn).
Lag optimisation: tests lags 0..MAX_LAGS for each predictor.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpi_nowcast.config import MODEL_CFG

logger = logging.getLogger(__name__)


class BridgeEquationModel:
    """
    Ridge / ElasticNet bridge equation with automatic lag selection.

    Parameters
    ----------
    regularisation : str
        'ridge' or 'elasticnet'.
    alpha : float, optional
        Regularisation strength.  Defaults to config.
    l1_ratio : float
        L1 ratio for ElasticNet (ignored for Ridge).
    max_lags : int
        Maximum lag (in months) to test for each predictor.
    n_cv_splits : int
        Number of time-series CV splits for lag selection.
    """

    def __init__(
        self,
        regularisation: str = "ridge",
        alpha: Optional[float] = None,
        l1_ratio: float = MODEL_CFG.elasticnet_l1_ratio,
        max_lags: int = MODEL_CFG.max_lags,
        n_cv_splits: int = 5,
    ) -> None:
        self.regularisation = regularisation
        self.alpha = alpha or (
            MODEL_CFG.ridge_alpha if regularisation == "ridge"
            else MODEL_CFG.elasticnet_alpha
        )
        self.l1_ratio = l1_ratio
        self.max_lags = max_lags
        self.n_cv_splits = n_cv_splits

        self._best_lags: Dict[str, int] = {}
        self._pipeline: Optional[Pipeline] = None
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lagged_features(
        self, X: pd.DataFrame, lags: Dict[str, int]
    ) -> pd.DataFrame:
        """Apply chosen lags to each predictor column."""
        cols: list[pd.Series] = []
        for col, lag in lags.items():
            if col not in X.columns:
                continue
            s = X[col].shift(lag)
            s.name = f"{col}_L{lag}"
            cols.append(s)
        if not cols:
            return pd.DataFrame(index=X.index)
        return pd.concat(cols, axis=1)

    def _select_lags(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, int]:
        """
        For each predictor, select the lag (0..max_lags) that minimises
        out-of-sample MSE in a time-series cross-validation.

        Returns
        -------
        dict mapping column name → best lag
        """
        best_lags: Dict[str, int] = {}
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        for col in X.columns:
            best_mse = np.inf
            best_lag = 0
            for lag in range(0, self.max_lags + 1):
                x_lag = X[[col]].shift(lag)
                df_cv = pd.concat([y, x_lag], axis=1).dropna()
                if len(df_cv) < 20:
                    continue
                y_cv = df_cv.iloc[:, 0].values
                X_cv = df_cv.iloc[:, 1:].values
                fold_mses: list[float] = []
                for train_idx, val_idx in tscv.split(X_cv):
                    if len(val_idx) == 0:
                        continue
                    reg = Ridge(alpha=self.alpha)
                    reg.fit(X_cv[train_idx], y_cv[train_idx])
                    preds = reg.predict(X_cv[val_idx])
                    fold_mses.append(np.mean((preds - y_cv[val_idx]) ** 2))
                if fold_mses:
                    avg_mse = np.mean(fold_mses)
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_lag = lag
            best_lags[col] = best_lag
        logger.debug("Lag selection result: %s", best_lags)
        return best_lags

    def _make_pipeline(self) -> Pipeline:
        if self.regularisation == "elasticnet":
            reg = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=5000,
                random_state=MODEL_CFG.random_seed,
            )
        else:
            reg = Ridge(alpha=self.alpha)
        return Pipeline([("scaler", StandardScaler()), ("reg", reg)])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BridgeEquationModel":
        """
        Fit the bridge equation model.

        Parameters
        ----------
        X : pd.DataFrame
            Monthly predictor matrix (YoY growth rates, winsorised).
        y : pd.Series
            Monthly CPI YoY growth rate (target).

        Returns
        -------
        self
        """
        logger.info("BridgeEquation: selecting lags for %d predictors…", X.shape[1])
        self._best_lags = self._select_lags(X, y)

        X_lagged = self._build_lagged_features(X, self._best_lags)
        df = pd.concat([y, X_lagged], axis=1).dropna()
        if df.empty:
            raise ValueError("No valid training rows after lag alignment and NaN removal.")

        y_train = df.iloc[:, 0].values
        X_train = df.iloc[:, 1:].values
        self._feature_names = df.columns[1:].tolist()

        self._pipeline = self._make_pipeline()
        self._pipeline.fit(X_train, y_train)
        logger.info(
            "BridgeEquation fitted on %d observations, %d features",
            len(y_train), X_train.shape[1],
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate nowcast predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as training X).

        Returns
        -------
        pd.Series
            Predicted CPI YoY; DatetimeIndex aligned with X.
        """
        if self._pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_lagged = self._build_lagged_features(X, self._best_lags)
        # Align columns
        X_lagged = X_lagged.reindex(columns=self._feature_names)
        # Only predict rows where all features are present
        mask = X_lagged.notna().all(axis=1)
        result = pd.Series(np.nan, index=X.index, name="bridge_nowcast")
        if mask.sum() == 0:
            logger.warning("BridgeEquation.predict: no complete feature rows")
            return result
        preds = self._pipeline.predict(X_lagged.loc[mask].values)
        result.loc[mask] = preds
        return result

    def feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose the prediction into per-feature contributions.

        Uses the linear structure: contribution_i = β_i * x_i (scaled).

        Returns
        -------
        pd.DataFrame
            Columns per feature; rows per time period.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not fitted.")

        scaler: StandardScaler = self._pipeline.named_steps["scaler"]
        reg = self._pipeline.named_steps["reg"]

        X_lagged = self._build_lagged_features(X, self._best_lags)
        X_lagged = X_lagged.reindex(columns=self._feature_names)
        mask = X_lagged.notna().all(axis=1)
        X_scaled = scaler.transform(X_lagged.loc[mask].fillna(0).values)

        coef = reg.coef_  # shape (n_features,)
        contributions = X_scaled * coef  # element-wise

        df = pd.DataFrame(
            contributions,
            index=X_lagged.loc[mask].index,
            columns=self._feature_names,
        )
        return df

    @property
    def coef_(self) -> pd.Series:
        """Regression coefficients as a named Series."""
        if self._pipeline is None:
            return pd.Series()
        return pd.Series(
            self._pipeline.named_steps["reg"].coef_,
            index=self._feature_names,
        )
