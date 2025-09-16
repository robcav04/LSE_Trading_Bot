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

logger = logging.getLogger("TradingBot")


class ModelHandler:
    """
    Loads the ML model and scaler, handles all feature engineering,
    and runs batch predictions. This module is the core of the trading strategy's logic.
    """

    # Names expected to be the first 4 entries in FEATURE_NAMES (order matters)
    _NORM_OHLC_NAMES: Tuple[str, str, str, str] = (
        "Open_norm_log",
        "High_norm_log",
        "Low_norm_log",
        "Close_norm_log",
    )

    def __init__(self, model_path: Path, scaler_path: Path):
        # ---- Validate config early for clearer failures ----
        if not hasattr(config, "FEATURE_NAMES") or not isinstance(config.FEATURE_NAMES, (list, tuple)):
            raise AttributeError("config.FEATURE_NAMES must be defined as a list/tuple.")
        if len(config.FEATURE_NAMES) < 5:
            raise ValueError("config.FEATURE_NAMES appears too short. Expected at least 5 features.")
        # Basic check that the first 4 features are the normalized OHLC names
        first4 = tuple(config.FEATURE_NAMES[:4])
        if first4 != self._NORM_OHLC_NAMES:
            logger.warning(
                f"First 4 FEATURE_NAMES {first4} do not match expected {self._NORM_OHLC_NAMES}. "
                "Ensure training and serving feature orders match exactly."
            )

        if not hasattr(config, "INDICATORS") or not isinstance(config.INDICATORS, (list, tuple)):
            raise AttributeError("config.INDICATORS must be defined (list of indicator configs).")
        if not hasattr(config, "WINDOW_SIZE") or not isinstance(config.WINDOW_SIZE, int) or config.WINDOW_SIZE <= 0:
            raise AttributeError("config.WINDOW_SIZE must be a positive integer.")

        self.feature_names: List[str] = list(config.FEATURE_NAMES)
        self.window: int = int(config.WINDOW_SIZE)

        # ---- Device selection ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelHandler using device: {self.device}")

        # ---- Load TorchScript model ----
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()  # Set model to evaluation mode
            logger.info(f"Successfully loaded TorchScript model from {model_path}")
        except Exception as e:
            logger.critical(f"Failed to load TorchScript model from {model_path}: {e}", exc_info=True)
            raise

        # ---- Load scaler ----
        scaler_path = Path(scaler_path)
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Successfully loaded scaler from {scaler_path}")
        except Exception as e:
            logger.critical(f"Failed to load scaler from {scaler_path}: {e}", exc_info=True)
            raise

        # Map indicator method names to functions
        self._indicator_map = {
            "rsi": rsi,
            "adx": adx,
            "bbands": bbands,
            "stoch": stoch,
            "roc": roc,
            "willr": willr,
        }

    # -----------------
    # Helper functions
    # -----------------

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators on a raw OHLCV DataFrame.
        Mirrors the training dataset process as closely as possible.

        Assumes df has at least columns: ["Open", "High", "Low", "Close", "Volume"].
        """
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for indicators: {missing}")

        out_df = df.copy()

        for idx, ind_cfg in enumerate(config.INDICATORS):
            try:
                method = ind_cfg["method"]
                params = ind_cfg.get("params", {}) or {}
                output_cols = list(ind_cfg.get("output_cols", []))
            except Exception as e:
                logger.warning(f"Indicator config at index {idx} is invalid: {e}. Skipping.")
                continue

            func = self._indicator_map.get(method)
            if func is None:
                logger.warning(f"Unknown indicator method '{method}'. Skipping.")
                continue

            try:
                # Many indicators use High/Low/Close; some may only need Close. Pass generously.
                res = func(high=out_df.get("High"), low=out_df.get("Low"), close=out_df.get("Close"), **params)

                # Normalize result assignment to match output_cols
                if isinstance(res, pd.DataFrame):
                    # If counts match, assign positionally; else try name-based mapping
                    if output_cols and len(output_cols) == res.shape[1]:
                        for j, col in enumerate(output_cols):
                            out_df[col] = res.iloc[:, j].astype("float32")
                    else:
                        # Best-effort: match by uppercased base name (e.g., 'ADX_14' -> 'ADX')
                        for col in output_cols:
                            base = col.split("_")[0].upper()
                            if base in res.columns:
                                out_df[col] = res[base].astype("float32")
                            else:
                                # If not found, try case-insensitive match
                                match = [c for c in res.columns if c.upper() == base]
                                if match:
                                    out_df[col] = res[match[0]].astype("float32")
                                else:
                                    out_df[col] = np.nan
                else:
                    # res is a Series or array-like; map to the first output column
                    target_col = output_cols[0] if output_cols else f"{method}_val"
                    out_df[target_col] = pd.Series(res, index=out_df.index, dtype="float32")

            except Exception as e:
                logger.warning(f"Indicator '{method}' failed: {e}. Filling NaNs for {output_cols}.")
                for col in output_cols:
                    out_df[col] = np.nan

        return out_df

    def _build_feature_matrix(
        self,
        df_with_indicators: pd.DataFrame,
        live_open_price: float,
    ) -> Optional[np.ndarray]:
        """
        Construct a (window, num_features) matrix for a single ticker,
        aligned exactly to self.feature_names.

        Returns None if data is insufficient or contains NaNs.
        """
        # Shift indicator columns by 1 to avoid lookahead, but do NOT shift the OHLC part
        indicator_cols = [c for c in self.feature_names if c not in self._NORM_OHLC_NAMES]
        present_indicator_cols = [c for c in indicator_cols if c in df_with_indicators.columns]
        missing_ind_cols = [c for c in indicator_cols if c not in present_indicator_cols]
        if missing_ind_cols:
            logger.debug(f"Missing expected indicator columns {missing_ind_cols}; will result in NaNs.")

        # Apply shift only on available indicator columns
        if present_indicator_cols:
            df_shifted = df_with_indicators.copy()
            df_shifted[present_indicator_cols] = df_shifted[present_indicator_cols].shift(1)
        else:
            df_shifted = df_with_indicators

        # Take last `window` rows
        df_window = df_shifted.tail(self.window).copy()

        # Basic validations
        ohlc_required = {"Open", "High", "Low", "Close"}
        if not ohlc_required.issubset(df_window.columns):
            logger.debug("OHLC columns missing in window slice.")
            return None
        if len(df_window) < self.window:
            return None

        # Normalize OHLC by live open price
        if not (isinstance(live_open_price, (int, float)) and np.isfinite(live_open_price) and live_open_price > 0):
            return None
        live_open_price = float(live_open_price)

        # Prepare matrix
        num_features = len(self.feature_names)
        mat = np.empty((self.window, num_features), dtype=np.float32)

        # Fill normalized OHLC part (must follow the exact order in _NORM_OHLC_NAMES)
        try:
            mat[:, 0] = np.log(df_window["Open"].to_numpy(dtype=np.float64) / live_open_price)
            mat[:, 1] = np.log(df_window["High"].to_numpy(dtype=np.float64) / live_open_price)
            mat[:, 2] = np.log(df_window["Low"].to_numpy(dtype=np.float64) / live_open_price)
            mat[:, 3] = np.log(df_window["Close"].to_numpy(dtype=np.float64) / live_open_price)
        except Exception as e:
            logger.debug(f"Failed to build normalized OHLC features: {e}")
            return None

        # Fill indicator part in the exact order of feature_names
        for j, feat in enumerate(self.feature_names[4:], start=4):
            if feat in df_window.columns:
                vals = pd.to_numeric(df_window[feat], errors="coerce").to_numpy(dtype=np.float32)
            else:
                # Missing indicator columns translate to NaNs
                vals = np.full((self.window,), np.nan, dtype=np.float32)
            mat[:, j] = vals

        # If any NaNs exist (from shifting or failed indicators), we skip this ticker
        if not np.isfinite(mat).all():
            return None

        return mat

    # -------------
    # Public API
    # -------------

    def predict(self, raw_historical_data: Dict[str, pd.DataFrame], open_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Takes raw historical data and live opening prices, performs all necessary
        feature engineering, and returns a dictionary of predictions.

        Args:
            raw_historical_data: Dict[ticker -> DataFrame] of last N days of raw OHLCV data.
            open_prices: Dict[ticker -> live opening price].

        Returns:
            Dict[ticker -> predicted log return] for tickers with valid features.
        """
        if not raw_historical_data or not open_prices:
            logger.warning("predict() called with empty raw_historical_data or open_prices.")
            return {}

        logger.info(f"Preparing feature batch for up to {len(open_prices)} stocks with live prices.")

        feature_mats: List[np.ndarray] = []
        valid_tickers: List[str] = []

        # Build per-ticker features
        for ticker, live_open in open_prices.items():
            df_raw = raw_historical_data.get(ticker)
            if df_raw is None or df_raw.empty:
                logger.debug(f"Skipping {ticker}: no historical data found.")
                continue

            # Ensure ascending index and numeric OHLCV
            try:
                df_raw = df_raw.sort_index()
                # Coerce OHLCV numeric
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    if col in df_raw.columns:
                        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
                if df_raw[["Open", "High", "Low", "Close"]].isna().any().any():
                    logger.debug(f"Skipping {ticker}: NaNs in OHLC columns.")
                    continue
            except Exception as e:
                logger.debug(f"Skipping {ticker}: failed to sanitize raw data ({e}).")
                continue

            try:
                df_with_inds = self._calculate_indicators(df_raw)
            except Exception as e:
                logger.debug(f"Skipping {ticker}: indicator calculation error ({e}).")
                continue

            mat = self._build_feature_matrix(df_with_inds, live_open_price=live_open)
            if mat is None:
                logger.debug(f"Skipping {ticker}: incomplete feature matrix.")
                continue

            feature_mats.append(mat)
            valid_tickers.append(ticker)

        if not valid_tickers:
            logger.warning("No valid feature matrices could be constructed.")
            return {}

        # Assemble batch
        try:
            batch_np = np.stack(feature_mats, axis=0)  # (N, window, F)
        except Exception as e:
            logger.critical(f"Failed to stack feature batch: {e}", exc_info=True)
            return {}

        n_samples, seq_len, n_features = batch_np.shape
        if seq_len != self.window or n_features != len(self.feature_names):
            logger.critical(
                f"Feature batch shape mismatch: got (N={n_samples}, T={seq_len}, F={n_features}), "
                f"expected T={self.window}, F={len(self.feature_names)}."
            )
            return {}

        # Scale: reshape to (N*T, F), transform, reshape back
        try:
            reshaped = batch_np.reshape(-1, n_features)
            scaled = self.scaler.transform(reshaped)
            final_batch = scaled.reshape(n_samples, seq_len, n_features).astype(np.float32, copy=False)
        except Exception as e:
            logger.critical(f"Scaler transform failed: {e}", exc_info=True)
            return {}

        # Inference
        try:
            with torch.inference_mode():
                batch_tensor = torch.from_numpy(final_batch).to(self.device, non_blocking=True)
                outputs = self.model(batch_tensor)
                # Support models returning tensors or tuples
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                outputs = outputs.squeeze(-1) if outputs.ndim > 1 and outputs.shape[-1] == 1 else outputs
                preds_np = outputs.detach().cpu().numpy().astype(float).reshape(-1)
        except Exception as e:
            logger.critical(f"Model inference failed: {e}", exc_info=True)
            return {}

        # Post-process: drop non-finite
        predictions: Dict[str, float] = {}
        for tkr, val in zip(valid_tickers, preds_np):
            if isinstance(val, (int, float)) and np.isfinite(val):
                predictions[tkr] = float(val)
            else:
                logger.debug(f"Dropping non-finite prediction for {tkr}: {val}")

        logger.info(f"Inference complete. Returned predictions for {len(predictions)}/{len(valid_tickers)} tickers.")
        return predictions
