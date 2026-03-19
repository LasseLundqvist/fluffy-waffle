"""
Configuration for CPI Nowcasting Pipeline — United Kingdom.

All API keys, model parameters, and variable definitions live here.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
REPORTS_DIR = ROOT_DIR / "output" / "reports"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API keys (override via environment variables)
# ---------------------------------------------------------------------------
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")
EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1"
ECB_BASE_URL = "https://data-api.ecb.europa.eu/service"
ONS_BASE_URL = "https://api.ons.gov.uk/v1"
FAO_BASE_URL = "https://www.fao.org/faostat/api/v1"

# ---------------------------------------------------------------------------
# Data collection parameters
# ---------------------------------------------------------------------------
HISTORY_START = "2010-01-01"   # Earliest date to fetch
BACKTEST_START = "2015-01-01"  # First forecast origin for backtest
MIN_TRAIN_MONTHS = 36          # Minimum training observations

# ---------------------------------------------------------------------------
# FRED series (UK-relevant)
# ---------------------------------------------------------------------------
FRED_SERIES: Dict[str, str] = {
    "brent_oil":        "DCOILBRENTEU",      # Brent crude (USD/barrel, daily)
    "natural_gas_ttf":  "PNGASEUUSDM",       # TTF natural gas (monthly proxy)
    "usd_gbp":          "DEXUSUK",           # USD per GBP (daily)
    "fertilizer":       "PFERTILIZERINDEXM", # IMF Fertilizer Price Index (monthly)
}

# ---------------------------------------------------------------------------
# ECB SDMX series (GBP-relevant)
# ---------------------------------------------------------------------------
ECB_SERIES: Dict[str, str] = {
    "eur_gbp": "EXR.D.GBP.EUR.SP00.A",               # GBP per EUR (daily)
    "usd_eur": "EXR.D.USD.EUR.SP00.A",               # USD per EUR (daily) → compute USD/GBP
    "ois_2y":  "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y", # EUR 2Y OIS
    "ois_5y":  "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y", # EUR 5Y OIS
}

# ---------------------------------------------------------------------------
# ONS (Office for National Statistics) timeseries
# Dataset: MM23 (Consumer Price Indices)
# ---------------------------------------------------------------------------
ONS_SERIES: Dict[str, str] = {
    "cpi_uk":  "D7BT",  # CPI All Items Index (2015=100)
    "cpih_uk": "L55O",  # CPIH All Items Index (2015=100)
}
ONS_DATASET = "MM23"

# ---------------------------------------------------------------------------
# Google Trends keywords (geo=GB, English)
# ---------------------------------------------------------------------------
GTRENDS_KEYWORDS: Dict[str, List[str]] = {
    "inflation": ["inflation", "cost of living", "price rise", "price increase"],
    "energy":    ["energy bills", "petrol price", "gas price"],
    "food":      ["food prices", "grocery prices", "supermarket prices"],
}
GTRENDS_GEO = "GB"
GTRENDS_TIMEFRAME = "today 12-m"
GTRENDS_PCA_VARIANCE = 0.85   # Keep components explaining ≥ 85 % variance

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Shared hyperparameter defaults."""
    ridge_alpha: float = 1.0
    elasticnet_alpha: float = 0.1
    elasticnet_l1_ratio: float = 0.5
    max_lags: int = 3                  # Lags 0–max_lags tested per variable
    midas_degree: int = 3              # Almon polynomial degree
    ensemble_method: str = "inv_rmse"  # "equal" or "inv_rmse"
    random_seed: int = 42


MODEL_CFG = ModelConfig()

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
ADF_SIGNIFICANCE = 0.05        # p-value threshold for stationarity

# ---------------------------------------------------------------------------
# Rate-limit / retry settings
# ---------------------------------------------------------------------------
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 2         # seconds (doubled each attempt)
GTRENDS_RATE_LIMIT = 5         # max requests/minute
REQUEST_TIMEOUT = 30           # seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
