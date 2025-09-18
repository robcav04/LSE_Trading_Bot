# model_handler.py
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import config
# Import the same indicator functions used in your dataset creation script
from custom_ta import rsi, adx, bbands, stoch, roc, willr

# Initialize a dedicated logger for this module
logger = logging.getLogger("TradingBot.ModelHandler")


class ModelHandler:
    """
    Loads the ML model and scaler, handles all feature engineering,
    and runs batch predictions. This module is the core of the trading strategy's logic.

    It ensures that the feature generation process at inference time is an exact
    mirror of the process used during the model's training phase.
    """

    # A constant tuple to ensure the first 4 features are always the normalized OHLC values.
    # This provides a single source of truth and simplifies validation.
    _NORM_OHLC_NAMES: Tuple[str, str, str, str] = (
        "Open_norm_log",
        "High_norm_log",
        "Low_norm_log",
        "Close_norm_log",
    )

    def __init__(self, model_path: Path, scaler_path: Path):
        """
        Initializes the ModelHandler by loading the model and scaler,
        and validating the configuration.

        Args:
            model_path (Path): The file path to the TorchScript model.
            scaler_path (Path): The file path to the saved scikit-learn scaler.
        """
        self._validate_config()

        self.feature_names: List[str] = list(config.FEATURE_NAMES)
        self.window: int = int(config.WINDOW_SIZE)

        # --- Device selection (CPU or GPU) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelHandler will use device: {self.device}")

        # --- Load and prepare the TorchScript model and the scaler ---
        self.model = self._load_torchscript_model(model_path)
        self.scaler = self._load_scaler(scaler_path)

        # Map indicator method names from the config to their actual functions for dynamic calling.
        self._indicator_map = {
            "rsi": rsi,
            "adx": adx,
            "bbands": bbands,
            "stoch": stoch,
            "roc": roc,
            "willr": willr,
        }

    def _validate_config(self):
        """Performs early, explicit validation of required settings from the config file."""
        logger.debug("Validating configuration for ModelHandler.")
        if not hasattr(config, "FEATURE_NAMES") or not isinstance(config.FEATURE_NAMES, (list, tuple)):
            raise AttributeError("config.FEATURE_NAMES must be defined as a list or tuple.")
        if len(config.FEATURE_NAMES) < len(self._NORM_OHLC_NAMES) + 1:
            raise ValueError(f"config.FEATURE_NAMES appears too short. Expected at least {len(self._NORM_OHLC_NAMES) + 1} features.")

        # Crucial check: Ensure the order of features matches the training setup.
        first_four_features = tuple(config.FEATURE_NAMES[:4])
        if first_four_features != self._NORM_OHLC_NAMES:
            logger.warning(
                f"First 4 FEATURE_NAMES {first_four_features} do not match expected {self._NORM_OHLC_NAMES}. "
                "This can lead to incorrect predictions if it deviates from the training configuration."
            )

        if not hasattr(config, "INDICATORS") or not isinstance(config.INDICATORS, (list, tuple)):
            raise AttributeError("config.INDICATORS must be defined as a list of indicator configurations.")
        if not hasattr(config, "WINDOW_SIZE") or not isinstance(config.WINDOW_SIZE, int) or config.WINDOW_SIZE <= 0:
            raise AttributeError("config.WINDOW_SIZE must be a positive integer.")
        logger.info("Configuration validated successfully.")

    def _load_torchscript_model(self, model_path: Path) -> torch.jit.ScriptModule:
        """Loads the TorchScript model and sets it to evaluation mode."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
            logger.info(f"Successfully loaded TorchScript model from {model_path}")
            return model
        except Exception as e:
            logger.critical(f"Failed to load TorchScript model from {model_path}: {e}", exc_info=True)
            raise

    def _load_scaler(self, scaler_path: Path) -> Any:
        """Loads the pre-fitted scikit-learn scaler object."""
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Successfully loaded scaler from {scaler_path}")
            # Basic validation of the loaded object
            if not hasattr(scaler, "transform"):
                raise TypeError("Loaded object from scaler_path does not have a 'transform' method.")
            return scaler
        except Exception as e:
            logger.critical(f"Failed to load scaler from {scaler_path}: {e}", exc_info=True)
            raise

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all technical indicators as defined in the config on a raw OHLCV DataFrame.
        This method is designed to perfectly mirror the indicator calculation process
        from the `rawtodatasetIBAPI.py` script.
        """
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns for indicator calculation: {missing}")

        out_df = df.copy()

        for idx, ind_cfg in enumerate(config.INDICATORS):
            try:
                method = ind_cfg["method"]
                params = ind_cfg.get("params", {}) or {}
                output_cols = list(ind_cfg.get("output_cols", []))
            except (KeyError, TypeError) as e:
                logger.warning(f"Indicator config at index {idx} is invalid: {e}. Skipping.")
                continue

            func = self._indicator_map.get(method)
            if func is None:
                logger.warning(f"Unknown indicator method '{method}' defined in config. Skipping.")
                continue

            try:
                # Call the indicator function with the necessary data columns.
                res = func(high=out_df.get("High"), low=out_df.get("Low"), close=out_df.get("Close"), **params)

                # Assign the results to the DataFrame, handling both single (Series)
                # and multiple (DataFrame) return values from the indicator functions.
                if isinstance(res, pd.DataFrame):
                    if output_cols and len(output_cols) == res.shape[1]:
                        # If column counts match, assign positionally for simplicity.
                        for i, col_name in enumerate(output_cols):
                            out_df[col_name] = res.iloc[:, i].astype("float32")
                    else:
                        # Fallback to a name-based mapping if counts differ.
                        for col_name in output_cols:
                            base_name = col_name.split("_")[0].upper()
                            matched_cols = [c for c in res.columns if c.upper() == base_name]
                            if matched_cols:
                                out_df[col_name] = res[matched_cols[0]].astype("float32")
                            else:
                                out_df[col_name] = np.nan
                elif isinstance(res, pd.Series):
                    if output_cols:
                        out_df[output_cols[0]] = pd.Series(res, index=out_df.index, dtype="float32")
                    else:
                         logger.warning(f"Indicator '{method}' returned a Series but no 'output_cols' are defined.")

            except Exception as e:
                logger.error(f"Indicator '{method}' failed during calculation: {e}. Filling with NaNs.", exc_info=True)
                for col in output_cols:
                    out_df[col] = np.nan

        return out_df

    def _build_feature_matrix(self, df_with_indicators: pd.DataFrame, live_open_price: float) -> Optional[np.ndarray]:
        """
        Constructs a single feature matrix (window_size, num_features) for one ticker.
        This includes normalizing OHLC data and aligning all features in the correct order.
        Returns None if the data is insufficient or contains NaN values after processing.
        """
        # 1. Shift indicator columns to prevent lookahead bias, just as done in training.
        indicator_cols = [c for c in self.feature_names if c not in self._NORM_OHLC_NAMES]
        df_shifted = df_with_indicators.copy()
        df_shifted[indicator_cols] = df_shifted[indicator_cols].shift(1)

        # 2. Select the final window of data needed for the model.
        df_window = df_shifted.tail(self.window)

        # 3. Validate the resulting window.
        if len(df_window) < self.window:
            logger.debug(f"Skipping ticker due to insufficient data rows after processing (have {len(df_window)}, need {self.window}).")
            return None
        if not (isinstance(live_open_price, (int, float)) and np.isfinite(live_open_price) and live_open_price > 0):
            logger.debug(f"Skipping ticker due to invalid live_open_price: {live_open_price}")
            return None

        # 4. Construct the final matrix in the exact feature order.
        num_features = len(self.feature_names)
        feature_matrix = np.full((self.window, num_features), np.nan, dtype=np.float32)

        # Fill normalized OHLC data.
        try:
            ohlc_data = df_window[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float64)
            feature_matrix[:, 0:4] = np.log(ohlc_data / live_open_price)
        except Exception as e:
            logger.error(f"Failed to build normalized OHLC features: {e}", exc_info=True)
            return None # Critical step, cannot proceed without OHLC.

        # Fill indicator data.
        for i, feat_name in enumerate(self.feature_names):
            if i < 4: continue # Skip the OHLC names
            if feat_name in df_window.columns:
                feature_matrix[:, i] = pd.to_numeric(df_window[feat_name], errors="coerce").to_numpy(dtype=np.float32)

        # 5. Final check for any NaN values.
        if not np.isfinite(feature_matrix).all():
            logger.debug(f"Skipping ticker due to NaN values in the final feature matrix.")
            return None

        return feature_matrix

    def predict(self, raw_historical_data: Dict[str, pd.DataFrame], open_prices: Dict[str, float]) -> Dict[str, float]:
        """
        The main public method. It orchestrates the entire prediction process for a batch of tickers.

        Args:
            raw_historical_data: A dictionary mapping tickers to their raw OHLCV historical data as DataFrames.
            open_prices: A dictionary mapping tickers to their live opening prices.

        Returns:
            A dictionary mapping tickers to their predicted log return. Tickers with invalid data are excluded.
        """
        if not raw_historical_data or not open_prices:
            logger.warning("predict() called with empty raw_historical_data or open_prices.")
            return {}

        logger.info(f"Preparing feature batch for {len(open_prices)} tickers with live prices.")

        feature_matrices: List[np.ndarray] = []
        valid_tickers: List[str] = []

        # --- Feature Engineering Loop ---
        for ticker, live_open in open_prices.items():
            df_raw = raw_historical_data.get(ticker)
            if df_raw is None or df_raw.empty:
                logger.debug(f"Skipping {ticker}: No historical data provided.")
                continue

            try:
                # Sanitize raw data: sort, ensure numeric, drop NaNs in core columns.
                df_sanitized = df_raw.sort_index()
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    if col in df_sanitized.columns:
                        df_sanitized[col] = pd.to_numeric(df_sanitized[col], errors="coerce")
                if df_sanitized[["Open", "High", "Low", "Close"]].isna().any().any():
                    logger.debug(f"Skipping {ticker}: Found NaNs in core OHLC columns.")
                    continue

                # Calculate indicators and build the final feature matrix.
                df_with_indicators = self._calculate_indicators(df_sanitized)
                matrix = self._build_feature_matrix(df_with_indicators, live_open_price=live_open)

                if matrix is not None:
                    feature_matrices.append(matrix)
                    valid_tickers.append(ticker)
                else:
                    logger.debug(f"Skipping {ticker}: Failed to build a valid feature matrix.")

            except Exception as e:
                logger.error(f"Skipping {ticker} due to an unexpected error during feature engineering: {e}", exc_info=True)
                continue

        if not valid_tickers:
            logger.warning("No valid feature matrices could be constructed for any ticker.")
            return {}

        # --- Batch Assembly and Scaling ---
        try:
            batch_np = np.stack(feature_matrices, axis=0)  # Shape: (N, window_size, num_features)
            n_samples, seq_len, n_features = batch_np.shape

            # Reshape for the scaler, which expects a 2D array.
            reshaped_batch = batch_np.reshape(-1, n_features)
            scaled_batch_reshaped = self.scaler.transform(reshaped_batch)

            # Reshape back to the original 3D structure for the model.
            final_batch = scaled_batch_reshaped.reshape(n_samples, seq_len, n_features).astype(np.float32)
        except Exception as e:
            logger.critical(f"Failed to assemble or scale the feature batch: {e}", exc_info=True)
            return {}

        # --- Model Inference ---
        try:
            with torch.inference_mode(): # More efficient than torch.no_grad()
                batch_tensor = torch.from_numpy(final_batch).to(self.device)
                outputs = self.model(batch_tensor)
                
                # Handle cases where the model might return a tuple (e.g., (output, attention_weights))
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                    
                # Squeeze the last dimension if it's 1, to get a 1D tensor of predictions.
                if outputs.ndim > 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                    
                preds_np = outputs.detach().cpu().numpy().flatten()
        except Exception as e:
            logger.critical(f"Model inference failed: {e}", exc_info=True)
            return {}

        # --- Post-processing and Return ---
        predictions: Dict[str, float] = {}
        for ticker, pred_value in zip(valid_tickers, preds_np):
            if np.isfinite(pred_value):
                predictions[ticker] = float(pred_value)
            else:
                logger.warning(f"Dropping non-finite prediction for {ticker}: {pred_value}")

        logger.info(f"Inference complete. Generated {len(predictions)} valid predictions out of {len(valid_tickers)} tickers.")
        return predictions
