"""Configure root logger for the CPI nowcast package."""
import logging
from cpi_nowcast.config import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()],
)
