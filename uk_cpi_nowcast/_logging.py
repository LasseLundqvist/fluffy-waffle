"""Configure root logger for the UK CPI nowcast package."""
import logging
from uk_cpi_nowcast.config import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()],
)
