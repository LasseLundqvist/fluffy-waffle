"""
Ensemble model combining Bridge Equation and MIDAS predictions.

Combination methods:
- 'equal': simple average of all constituent forecasts
- 'inv_rmse': inverse-RMSE-weighted average (better performing models get more weight)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cpi_nowcast.config import MODEL_CFG

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Weighted ensemble of nowcast models.

    Parameters
    ----------
    method : str
        'equal' or 'inv_rmse'.
    window : int
        Rolling window (months) for computing out-of-sample RMSE weights.
        If None, uses the full history.
    """

    def __init__(
        self,
        method: str = MODEL_CFG.ensemble_method,
        window: Optional[int] = 24,
    ) -> None:
        self.method = method
        self.window = window
        self._weights: Dict[str, float] = {}
        self._model_names: List[str] = []

    def compute_weights(
        self,
        predictions: pd.DataFrame,
        actuals: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute combination weights from historical forecast errors.

        Parameters
        ----------
        predictions : pd.DataFrame
            Columns are model names; index is DatetimeIndex.
            Each column contains that model's point forecasts.
        actuals : pd.Series
            Realised CPI YoY values (DatetimeIndex).

        Returns
        -------
        dict
            Map from model name to weight (sum to 1).
        """
        if self.method == "equal":
            n = len(predictions.columns)
            return {col: 1.0 / n for col in predictions.columns}

        # inv_rmse weighting
        aligned = predictions.join(actuals.rename("actual"), how="inner").dropna()
        if aligned.empty:
            logger.warning("Ensemble: no aligned data for weight computation; using equal weights")
            n = len(predictions.columns)
            return {col: 1.0 / n for col in predictions.columns}

        if self.window is not None:
            aligned = aligned.iloc[-self.window:]

        rmse_map: Dict[str, float] = {}
        for col in predictions.columns:
            errors = aligned[col] - aligned["actual"]
            rmse = np.sqrt(np.mean(errors ** 2))
            rmse_map[col] = max(rmse, 1e-6)   # avoid division by zero

        inv_rmse = {col: 1.0 / v for col, v in rmse_map.items()}
        total = sum(inv_rmse.values())
        weights = {col: v / total for col, v in inv_rmse.items()}
        logger.info("Ensemble weights: %s", weights)
        return weights

    def fit(
        self,
        predictions: pd.DataFrame,
        actuals: pd.Series,
    ) -> "EnsembleModel":
        """
        Learn combination weights from historical predictions and actuals.

        Parameters
        ----------
        predictions : pd.DataFrame
            Historical point forecasts from constituent models.
        actuals : pd.Series
            Realised target values.
        """
        self._model_names = predictions.columns.tolist()
        self._weights = self.compute_weights(predictions, actuals)
        return self

    def predict(self, predictions: pd.DataFrame) -> pd.Series:
        """
        Combine predictions using the learned weights.

        Parameters
        ----------
        predictions : pd.DataFrame
            Current point forecasts from constituent models.
            Must include all columns seen at fit time.

        Returns
        -------
        pd.Series
            Ensemble nowcast.
        """
        if not self._weights:
            # Fallback to equal weights
            n = len(predictions.columns)
            weights = {col: 1.0 / n for col in predictions.columns}
        else:
            weights = self._weights

        aligned = predictions.reindex(columns=list(weights.keys()))
        ensemble = sum(aligned[col] * w for col, w in weights.items() if col in aligned.columns)
        if isinstance(ensemble, pd.Series):
            ensemble.name = "ensemble_nowcast"
        return ensemble

    def predict_with_intervals(
        self,
        predictions: pd.DataFrame,
        rmse_history: Optional[float] = None,
        n_sigma: float = 1.96,
    ) -> pd.DataFrame:
        """
        Produce point forecast + confidence intervals.

        The interval is derived from:
        1. Ensemble spread (std across models)
        2. Historical RMSE (if provided)

        Parameters
        ----------
        predictions : pd.DataFrame
            Constituent model predictions.
        rmse_history : float, optional
            Historical RMSE from backtest evaluation.
        n_sigma : float
            Number of standard deviations for the interval.

        Returns
        -------
        pd.DataFrame
            Columns: ensemble_nowcast, lower_68, upper_68, lower_95, upper_95.
        """
        point = self.predict(predictions)
        spread = predictions.std(axis=1)

        if rmse_history is not None:
            uncertainty = np.sqrt(spread ** 2 + rmse_history ** 2)
        else:
            uncertainty = spread.clip(lower=0.001)

        df = pd.DataFrame({"ensemble_nowcast": point})
        df["lower_68"] = point - 1.0 * uncertainty
        df["upper_68"] = point + 1.0 * uncertainty
        df["lower_95"] = point - n_sigma * uncertainty
        df["upper_95"] = point + n_sigma * uncertainty
        return df
