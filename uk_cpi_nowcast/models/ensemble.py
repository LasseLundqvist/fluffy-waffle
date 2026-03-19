"""
Ensemble model combining Bridge Equation and MIDAS predictions.

Combination methods:
- 'equal':    simple average
- 'inv_rmse': inverse-RMSE-weighted average (better models get higher weight)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from uk_cpi_nowcast.config import MODEL_CFG

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Weighted ensemble of nowcast models.

    Parameters
    ----------
    method : str
        'equal' or 'inv_rmse'.
    window : int, optional
        Rolling window (months) for RMSE weight computation.
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
        if self.method == "equal":
            n = len(predictions.columns)
            return {col: 1.0 / n for col in predictions.columns}

        aligned = predictions.join(actuals.rename("actual"), how="inner").dropna()
        if aligned.empty:
            logger.warning("Ensemble: no aligned data; using equal weights")
            n = len(predictions.columns)
            return {col: 1.0 / n for col in predictions.columns}

        if self.window is not None:
            aligned = aligned.iloc[-self.window:]

        rmse_map: Dict[str, float] = {}
        for col in predictions.columns:
            errors = aligned[col] - aligned["actual"]
            rmse = np.sqrt(np.mean(errors ** 2))
            rmse_map[col] = max(rmse, 1e-6)

        inv_rmse = {col: 1.0 / v for col, v in rmse_map.items()}
        total = sum(inv_rmse.values())
        weights = {col: v / total for col, v in inv_rmse.items()}
        logger.info("Ensemble weights: %s", weights)
        return weights

    def fit(self, predictions: pd.DataFrame, actuals: pd.Series) -> "EnsembleModel":
        self._model_names = predictions.columns.tolist()
        self._weights = self.compute_weights(predictions, actuals)
        return self

    def predict(self, predictions: pd.DataFrame) -> pd.Series:
        if not self._weights:
            n = len(predictions.columns)
            weights = {col: 1.0 / n for col in predictions.columns}
        else:
            weights = self._weights

        aligned = predictions.reindex(columns=list(weights.keys()))
        ensemble = sum(
            aligned[col] * w for col, w in weights.items() if col in aligned.columns
        )
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
        Produce point forecast + 68% and 95% confidence intervals.

        Uncertainty = ensemble spread + historical RMSE (combined in quadrature).
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
