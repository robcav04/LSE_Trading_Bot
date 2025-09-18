# utils.py
import logging
import sys
from pathlib import Path
from datetime import time

import pandas as pd
import pandas_market_calendars as mcal

import config

def setup_logging(log_dir: Path):
    """Configures logging to file and console."""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"bot_log_{pd.Timestamp.now().strftime('%Y%m%d')}.log"
    
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(sh)
    
    return logger

def get_lse_calendar():
    """Returns the LSE market calendar instance."""
    return mcal.get_calendar(config.MARKET_CALENDAR)

def get_market_schedule_for_date(check_date: pd.Timestamp, calendar):
    """Gets the open and close times for a specific date."""
    schedule = calendar.schedule(start_date=check_date, end_date=check_date)
    if not schedule.empty:
        return {
            "open": schedule.iloc[0]['market_open'].tz_convert(config.TIMEZONE),
            "close": schedule.iloc[0]['market_close'].tz_convert(config.TIMEZONE)
        }
    return None
