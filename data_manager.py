# data_manager.py
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import config

logger = logging.getLogger("TradingBot")

REQUIRED_OHLCV_COLS: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


class DataManager:
    """
    Handles reading from and writing to the local HDF5 database of raw
    historical stock data. It provides the necessary data for both live
    trading and overnight updates.
    """

    def __init__(self, db_path: Path = Path(getattr(config, "RAW_DATABASE_PATH", ""))):
        """
        Initializes the DataManager.

        Args:
            db_path (Path): The path to the HDF5 file containing raw OHLCV data.
        """
        if not db_path:
             raise ValueError("config.RAW_DATABASE_PATH is not defined or is empty.")
        self.db_path = Path(db_path)

        if not hasattr(config, "WINDOW_SIZE") or not isinstance(config.WINDOW_SIZE, int) or config.WINDOW_SIZE <= 0:
            raise ValueError("config.WINDOW_SIZE must be a positive integer.")

        logger.info(f"DataManager initialized for raw database: {self.db_path}")

    def _sanitize_key(self, key: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", key)

    def _validate_ohlcv_columns(self, df: pd.DataFrame, ticker: str) -> bool:
        missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
        if missing:
            logger.warning(f"{ticker}: missing required columns {missing}.")
            return False
        return True

    def _coerce_numeric(self, df: pd.DataFrame, cols: List[str], ticker: str) -> pd.DataFrame:
        df_copy = df.copy()
        for c in cols:
            df_copy[c] = pd.to_numeric(df_copy[c], errors="coerce")
        before = len(df_copy)
        df_copy = df_copy.dropna(subset=cols)
        after = len(df_copy)
        if after < before:
            logger.debug(f"{ticker}: dropped {before - after} rows with non-numeric OHLCV.")
        return df_copy

    def _select_last_n_rows(self, store: pd.HDFStore, key: str, n: int, ticker_for_log: str) -> Optional[pd.DataFrame]:
        """Safely reads the last N rows for a given key, with fallbacks."""
        try:
            # This is the most efficient method
            return store.select(key, start=-n)
        except (KeyError, IndexError):
             # Fallback for stores that don't support efficient slicing
            try:
                df_full = store.get(key)
                return df_full.tail(n)
            except Exception as e:
                logger.error(f"Could not read data for {ticker_for_log}: {e}")
        return None

    def get_all_latest_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads the most recent 'RAW_DATA_HISTORY_DAYS' data points of raw
        OHLCV data for all stocks available in the HDF5 file.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Cannot load data; raw database file does not exist at {self.db_path}")

        history_needed = int(config.RAW_DATA_HISTORY_DAYS)
        # Ensure enough data for the longest indicator (e.g., 20) plus the model window size
        min_required_rows = config.WINDOW_SIZE + 20 
        logger.info(f"Loading latest {history_needed}-day raw data for all stocks (requiring at least {min_required_rows} rows)...")
        all_latest_data: Dict[str, pd.DataFrame] = {}

        try:
            with pd.HDFStore(self.db_path, mode="r") as store:
                keys = [k.strip("/") for k in store.keys()]
                for ticker in tqdm(keys, desc="Loading Raw Historical Data"):
                    df = self._select_last_n_rows(store, ticker, history_needed, ticker_for_log=ticker)
                    
                    if df is None or df.empty:
                        continue
                    
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index, errors="coerce")
                        df.dropna(axis=0, how='any', subset=[df.index.name], inplace=True)
                    
                    if not self._validate_ohlcv_columns(df, ticker):
                        continue
                        
                    df = self._coerce_numeric(df, list(REQUIRED_OHLCV_COLS), ticker)
                    df = df.dropna(subset=list(REQUIRED_OHLCV_COLS))

                    if len(df) < min_required_rows:
                        logger.warning(f"Skipping {ticker}: has only {len(df)} valid rows, not enough for feature generation (min: {min_required_rows}).")
                        continue
                    
                    all_latest_data[ticker] = df.loc[:, list(REQUIRED_OHLCV_COLS)].sort_index()

            logger.info(f"Successfully loaded raw historical data for {len(all_latest_data)} stocks.")
            return all_latest_data
        except Exception as e:
            logger.critical(f"Failed to open or process raw HDF5 database: {e}", exc_info=True)
            return {}

    def update_raw_database(self, bars_to_update: Dict[str, dict]):
        """
        Appends new daily bars and removes the oldest to maintain a constant
        rolling window of historical data in the HDF5 database.
        """
        if not bars_to_update:
            logger.warning("No new bars provided to update_raw_database.")
            return

        if not self.db_path.exists():
            logger.error(f"Cannot update; raw database file does not exist at {self.db_path}. Run the initialisation script first.")
            return

        logger.info(f"Updating raw database with {len(bars_to_update)} new bars...")
        
        updated_count = 0
        try:
            with pd.HDFStore(self.db_path, mode="a") as store:
                for ticker, bar_data in bars_to_update.items():
                    safe_key = self._sanitize_key(ticker)
                    try:
                        df_new = pd.DataFrame([bar_data])
                        df_new["Date"] = pd.to_datetime(df_new["Date"], format="%Y%m%d")
                        df_new.set_index("Date", inplace=True)

                        df_old = store.get(safe_key)
                        df_combined = pd.concat([df_old, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
                        
                        # Keep only the most recent N days
                        df_rolled = df_combined.tail(config.RAW_DATA_HISTORY_DAYS)

                        store.put(safe_key, df_rolled, format="table", data_columns=True)
                        updated_count += 1
                    except Exception as e:
                        logger.error(f"Failed to update rolling window for ticker {ticker}: {e}")
            
            logger.info(f"Successfully updated the raw database for {updated_count} tickers.")
        except Exception as e:
            logger.critical(f"A critical error occurred while writing to the HDF5 database: {e}", exc_info=True)
    
    def seed_initial_database(self, initial_data: Dict[str, pd.DataFrame]):
        """
        Creates a new HDF5 database from a dictionary of DataFrames.
        This DELETES any existing file at the path.
        """
        if not initial_data:
            logger.error("No data provided to seed the database. Aborting.")
            return

        logger.info(f"Seeding new raw database at {self.db_path}. This will overwrite any existing file.")
        
        if self.db_path.exists():
            try:
                self.db_path.unlink()
                logger.info(f"Removed existing database file: {self.db_path}")
            except OSError as e:
                logger.critical(f"Could not remove existing database file: {e}. Aborting.")
                return

        try:
            with pd.HDFStore(self.db_path, mode="w") as store:
                for ticker, df in tqdm(initial_data.items(), desc="Seeding Database"):
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        logger.warning(f"Skipping {ticker}: invalid or empty DataFrame provided.")
                        continue
                    
                    safe_key = self._sanitize_key(ticker)
                    store.put(safe_key, df, format='table', data_columns=True)
            logger.info(f"Database seeding complete. Saved data for {len(initial_data)} tickers.")
        except Exception as e:
            logger.critical(f"A critical error occurred during database seeding: {e}", exc_info=True)
