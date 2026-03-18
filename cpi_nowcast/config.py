"""
Configuration for CPI Nowcasting Pipeline — Denmark.

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
DST_BASE_URL = "https://api.statbank.dk/v1"
FAO_BASE_URL = "https://www.fao.org/faostat/api/v1"
ENTSO_E_BASE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSO_E_TOKEN: str = os.environ.get("ENTSO_E_TOKEN", "")

# ---------------------------------------------------------------------------
# Data collection parameters
# ---------------------------------------------------------------------------
HISTORY_START = "2010-01-01"   # Earliest date to fetch
BACKTEST_START = "2015-01-01"  # First forecast origin for backtest
MIN_TRAIN_MONTHS = 36          # Minimum training observations

# ---------------------------------------------------------------------------
# FRED series
# ---------------------------------------------------------------------------
FRED_SERIES: Dict[str, str] = {
    "brent_oil": "DCOILBRENTEU",        # Brent crude (USD/barrel, daily)
    "natural_gas_ttf": "PNGASEUUSDM",   # TTF natural gas (monthly proxy)
    "bdi": "BOGZ1FL663067003Q",         # Baltic Dry Index (quarterly, fallback)
}

# ---------------------------------------------------------------------------
# ECB SDMX series
# ---------------------------------------------------------------------------
ECB_SERIES: Dict[str, str] = {
    "usd_dkk": "EXR.D.DKK.USD.SP00.A",
    "eur_dkk": "EXR.D.DKK.EUR.SP00.A",
    "ois_2y":  "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",   # 2Y OIS
    "ois_5y":  "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y",   # 5Y OIS
}

# ---------------------------------------------------------------------------
# DST (Danmarks Statistik) tables
# ---------------------------------------------------------------------------
DST_TABLES: Dict[str, str] = {
    "cpi":              "PRIS111",   # Danish CPI (national)
    "hicp":             "PRIS114",   # Danish HICP
    "consumer_conf":    "FORV1",     # Consumer confidence (monthly)
}

# ---------------------------------------------------------------------------
# Google Trends keywords (geo=DK)
# ---------------------------------------------------------------------------
GTRENDS_KEYWORDS: Dict[str, List[str]] = {
    "inflation": ["prisstigninger", "inflation", "dyrere", "prisstigning"],
    "energy":    ["elpris", "benzinpris", "varmepris"],
    "food":      ["madpriser", "dagligvarer dyrere"],
}
GTRENDS_GEO = "DK"
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
