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

        # For seeding, the file doesn't need to exist yet.
        # For other operations, we'll check existence as needed.

        if not hasattr(config, "WINDOW_SIZE") or not isinstance(config.WINDOW_SIZE, int) or config.WINDOW_SIZE <= 0:
            raise ValueError("config.WINDOW_SIZE must be a positive integer.")

        logger.info(f"DataManager initialized for raw database: {self.db_path}")

    # --------------------------
    # Internal helper utilities
    # (Your helper functions were excellent and are preserved here without changes)
    # --------------------------
    def _sanitize_key(self, key: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", key)

    def _parse_date_value(self, value) -> pd.Timestamp:
        s = str(value).strip()
        if len(s) == 8 and s.isdigit():
            try:
                return pd.to_datetime(s, format="%Y%m%d")
            except Exception:
                pass
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Unparseable date: {value!r}")
        return ts

    def _validate_ohlcv_columns(self, df: pd.DataFrame, ticker: str) -> bool:
        missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
        if missing:
            logger.warning(f"{ticker}: missing required columns {missing}.")
            return False
        return True

    def _coerce_numeric(self, df: pd.DataFrame, cols: List[str], ticker: str) -> pd.DataFrame:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        before = len(df)
        df = df.dropna(subset=cols)
        after = len(df)
        if after < before:
            logger.debug(f"{ticker}: dropped {before - after} rows with non-numeric OHLCV.")
        return df

    def _select_last_n_rows(self, store: pd.HDFStore, key: str, n: int, ticker_for_log: str) -> Optional[pd.DataFrame]:
        try:
            return store.select(key, start=-n)
        except Exception:
            try:
                df_full = store.get(key)
                return df_full.tail(n)
            except Exception as e2:
                logger.error(f"{ticker_for_log}: fallback read failed: {e2}")
        return None
    
    # --------------------------
    # Public methods
    # --------------------------


    def get_all_latest_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads the most recent 'RAW_DATA_HISTORY_DAYS' data points of raw
        OHLCV data for all stocks available in the HDF5 file.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Cannot load data; raw database file does not exist at {self.db_path}")

        history_needed = int(config.RAW_DATA_HISTORY_DAYS)
        logger.info(f"Loading latest {history_needed}-day raw data for all stocks...")
        all_latest_data: Dict[str, pd.DataFrame] = {}

        try:
            with pd.HDFStore(self.db_path, mode="r") as store:
                keys = [k.strip("/") for k in store.keys()]
                for ticker in tqdm(keys, desc="Loading Raw Historical Data"):
                    # Use the corrected history_needed variable here
                    df = self._select_last_n_rows(store, ticker, history_needed, ticker_for_log=ticker)
                    
                    if df is None or df.empty:
                        continue

                    # The rest of the validation logic remains the same
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index, errors="coerce")
                        if df.index.isna().any():
                            logger.warning(f"Skipping {ticker}: invalid/non-datetime index.")
                            continue
                    
                    if not self._validate_ohlcv_columns(df, ticker):
                        continue
                    df = self._coerce_numeric(df, list(REQUIRED_OHLCV_COLS), ticker)

                    # Ensure we have at least enough data for the model handler to work with
                    if len(df) < (config.WINDOW_SIZE + 20): # 20 is for longest indicator
                        logger.warning(f"Skipping {ticker}: has only {len(df)} valid rows, not enough for feature generation.")
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

        logger.info(f"Updating raw database with {len(bars_to_update)} new bars, maintaining a {config.RAW_DATA_HISTORY_DAYS}-day window.")
        
        updated_count = 0
        try:
            # Use mode="a" to read and write to the existing file
            with pd.HDFStore(self.db_path, mode="a") as store:
                for ticker, bar_data in bars_to_update.items():
                    safe_key = self._sanitize_key(ticker)
                    try:
                        # Create a DataFrame for the new bar
                        df_new = pd.DataFrame([bar_data])
                        df_new["Date"] = pd.to_datetime(df_new["Date"], format="%Y%m%d")
                        df_new.set_index("Date", inplace=True)

                        # Read the existing data for the ticker
                        df_old = store.get(safe_key)

                        # Combine old and new data
                        df_combined = pd.concat([df_old, df_new])

                        # Remove duplicates and sort by date to be safe
                        df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()

                        # Keep only the most recent N days
                        df_rolled = df_combined.tail(config.RAW_DATA_HISTORY_DAYS)

                        # Overwrite the old data with the new rolling window
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
        This DELETES any existing file at the path. To be run only once for setup.
        """
        if not initial_data:
            logger.error("No data provided to seed the database. Aborting.")
            return

        logger.info(f"Seeding new raw database at {self.db_path}. This will overwrite any existing file.")
        
        # Delete existing file to ensure a clean start
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
