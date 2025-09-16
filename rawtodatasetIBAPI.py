from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import joblib
import numba as nb
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1  Configuration & defaults
# -----------------------------------------------------------------------------

class IndicatorConfig(BaseModel):
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)
    output_cols: List[str]


class BuilderConfig(BaseModel):
    input_hdf5_path: Path
    output_hdf5_file_with_splits: Path
    window_size: int = Field(14, ge=2)
    target_offset: int = Field(1, ge=1)
    min_rows: int = Field(60, ge=10)
    indicators: List[IndicatorConfig]
    num_workers: int = Field(max(mp.cpu_count() // 2, 1), ge=1)
    data_start_cutoff: Optional[str] = "2010-01-01"
    strict_window_split: bool = True
    debug_checks: bool = False
    debug_single_stock: bool = False # New flag for debugging
    features_to_transform_with_asinh: List[str] = Field(default_factory=list)
    standardize_all_features: bool = True
    scaler_save_path: Optional[Path] = None

    @field_validator("input_hdf5_path", "output_hdf5_file_with_splits", mode="before")
    @classmethod
    def _expand(cls, v, info):
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def derive_paths(self) -> "BuilderConfig":
        if self.scaler_save_path is None and self.output_hdf5_file_with_splits:
            stem = self.output_hdf5_file_with_splits.stem
            self.scaler_save_path = self.output_hdf5_file_with_splits.parent / f"{stem}_scaler.joblib"
        return self

# -----------------------------------------------------------------------------
# 2  Technical Indicators (from custom_ta.py)
# -----------------------------------------------------------------------------

def _wilder_ema(series: pd.Series, length: int) -> pd.Series:
    """Calculates the Wilder's Exponential Moving Average."""
    return series.ewm(com=length - 1, adjust=False, min_periods=length).mean()

def rsi(close: pd.Series, length: int = 14, **kwargs) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = _wilder_ema(gain, length)
    avg_loss = _wilder_ema(loss, length)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, **kwargs):
    """Average Directional Movement Index (ADX)"""
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = _wilder_ema(tr, length)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    plus_di = 100 * _wilder_ema(plus_dm, length) / atr
    minus_di = 100 * _wilder_ema(minus_dm, length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = _wilder_ema(dx.fillna(0), length)
    return pd.DataFrame({'ADX': adx_series, 'DMP': plus_di})

def bbands(close: pd.Series, length: int = 20, std: float = 2.0, **kwargs):
    """Bollinger Bands"""
    sma = close.rolling(length, min_periods=length).mean()
    rolling_std = close.rolling(length, min_periods=length).std(ddof=0)
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    bbb = 100 * (upper - lower) / sma
    return pd.DataFrame({'BBB': bbb})

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3, smooth_k: int = 3, **kwargs):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = k_line.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(d).mean()
    return pd.DataFrame({'STOCHK': stoch_k, 'STOCHD': stoch_d})

def roc(close: pd.Series, length: int = 12, **kwargs) -> pd.Series:
    """Rate of Change (ROC)"""
    return 100 * (close.diff(length) / close.shift(length))

def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, **kwargs) -> pd.Series:
    """Williams %R"""
    lowest_low = low.rolling(length).min()
    highest_high = high.rolling(length).max()
    willr_series = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return willr_series

# -----------------------------------------------------------------------------
# 3  Low-level helpers
# -----------------------------------------------------------------------------

COLS_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

@nb.njit(cache=True, fastmath=True)
def _asinh(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        v = x[i]
        out[i] = np.log(v + np.sqrt(v * v + 1.0))
    return out

@nb.njit(cache=True, fastmath=True)
def window_to_matrix(open_a, high_a, low_a, close_a, ind_a, tgt_open):
    ws, n_ind = open_a.shape[0], ind_a.shape[1]
    out = np.empty((ws, 4 + n_ind), dtype=np.float32)
    tgt_open = max(tgt_open, 1e-9)
    for t in range(ws):
        out[t, 0] = np.log(max(open_a[t], 1e-6) / tgt_open)
        out[t, 1] = np.log(max(high_a[t], 1e-6) / tgt_open)
        out[t, 2] = np.log(max(low_a[t], 1e-6) / tgt_open)
        out[t, 3] = np.log(max(close_a[t], 1e-6) / tgt_open)
        for j in range(n_ind):
            out[t, 4 + j] = ind_a[t, j]
    return out

# -----------------------------------------------------------------------------
# 4  Per-stock window builder
# -----------------------------------------------------------------------------

class StockBuilder:
    def __init__(self, cfg: BuilderConfig, ind_cols: List[str]):
        self.cfg = cfg
        self.ind_cols = ind_cols
        self.required_cols = COLS_OHLCV + ind_cols

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.cfg.indicators:
            return out
        indicator_map = {
            "rsi": rsi, "adx": adx, "bbands": bbands,
            "stoch": stoch, "roc": roc, "willr": willr
        }
        for ind_cfg in self.cfg.indicators:
            try:
                indicator_func = indicator_map.get(ind_cfg.method)
                if indicator_func:
                    res = indicator_func(
                        high=out.get("High"), low=out.get("Low"),
                        close=out["Close"], **ind_cfg.params
                    )
                    if isinstance(res, pd.DataFrame):
                        for col in ind_cfg.output_cols:
                            out[col] = res[col.split('_')[0].upper()]
                    else: # Series
                        out[ind_cfg.output_cols[0]] = res
            except Exception as exc:
                logging.debug("Custom TA fail for %s - %s", ind_cfg.method, exc)
                for col in ind_cfg.output_cols:
                    if col not in out:
                        out[col] = np.nan
        return out

    @staticmethod
    def _shift_back(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df2 = df.copy()
        df2[cols] = df2[cols].shift(1)
        return df2

    def __call__(self, df: pd.DataFrame, sid: str):
        if self.cfg.debug_single_stock:
            logging.info(f"[{sid}] Initial df shape: {df.shape}")

        if df.shape[0] < self.cfg.min_rows:
            if self.cfg.debug_single_stock:
                logging.info(f"[{sid}] SKIPPED: Not enough rows ({df.shape[0]} < {self.cfg.min_rows})")
            return []

        df = df[COLS_OHLCV].apply(pd.to_numeric, errors="coerce").dropna()
        if self.cfg.debug_single_stock:
            logging.info(f"[{sid}] After OHLCV selection & dropna: {df.shape}")

        df = df[(df[COLS_OHLCV[:-1]] > 1e-6).all(axis=1) & (df["Volume"] > 1e-6)]
        df = df.sort_index()
        if self.cfg.debug_single_stock:
            logging.info(f"[{sid}] After filtering low values: {df.shape}")

        df = self._add_indicators(df)
        if self.cfg.debug_single_stock:
            logging.info(f"[{sid}] After adding indicators: {df.shape}")
            # Log NaN counts per indicator column to see what's being generated
            nan_counts = df[self.ind_cols].isnull().sum()
            logging.info(f"[{sid}] NaN counts in indicator columns:\n{nan_counts}")


        df = self._shift_back(df, self.ind_cols)
        for col in self.ind_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df.dropna(subset=self.required_cols)
        if self.cfg.debug_single_stock:
            logging.info(f"[{sid}] After final dropna on required columns: {df.shape}")

        if df.empty:
            if self.cfg.debug_single_stock:
                logging.info(f"[{sid}] SKIPPED: DataFrame is empty after cleaning.")
            return []

        open_a, high_a, low_a, close_a, _ = [df[c].values.astype(np.float32) for c in COLS_OHLCV]
        ind_arr = df[self.ind_cols].values.astype(np.float32)
        n, ws, off = df.shape[0], self.cfg.window_size, self.cfg.target_offset
        if n < ws + off:
            return []

        mats, tgts, dates = [], [], []
        for i in range(n - ws - off + 1):
            tgt_idx = i + ws + off - 1
            tgt_row = df.iloc[tgt_idx]
            O_t, C_t = float(tgt_row["Open"]), float(tgt_row["Close"])
            if O_t <= 1e-6 or C_t <= 1e-6:
                continue
            mat = window_to_matrix(open_a[i:i+ws], high_a[i:i+ws], low_a[i:i+ws], close_a[i:i+ws], ind_arr[i:i+ws], O_t)
            mats.append(mat)
            tgts.append(np.log(C_t / O_t))
            dates.append(tgt_row.name.strftime("%Y-%m-%d"))

        if not mats:
            return []

        return [{"features": np.stack(mats), "targets": np.array(tgts, dtype=np.float32), "dates": np.array(dates, dtype="S10"), "stock": sid}]


# -----------------------------------------------------------------------------
# 5  Helper functions
# -----------------------------------------------------------------------------

def derive_indicator_cols(cfg: BuilderConfig) -> List[str]:
    """Derives the full list of indicator column names from config."""
    if not cfg.indicators:
        return []
    ind_cols = [c for ind in cfg.indicators for c in ind.output_cols]
    return list(dict.fromkeys(ind_cols))

def loss_weights(date_strings: List[str], tr_min: Optional[str], tr_max: Optional[str]) -> np.ndarray:
    if not date_strings or tr_min is None or tr_max is None:
        return np.ones(len(date_strings), dtype=np.float32)
    dmin, dmax = pd.Timestamp(tr_min), pd.Timestamp(tr_max)
    span = (dmax - dmin).days
    if span <= 0:
        return np.ones(len(date_strings), dtype=np.float32)
    ts = pd.to_datetime(date_strings)
    exps = (dmax - ts).days / span
    return np.power(0.5, exps.astype(np.float32))

# -----------------------------------------------------------------------------
# 6  Data Loading
# -----------------------------------------------------------------------------

def load_data_from_hdf5(fp: Path) -> Dict[str, pd.DataFrame]:
    """Loads all stock DataFrames from the HDF5 file."""
    logging.info("Loading data from %s", fp)
    all_stocks = {}
    try:
        with pd.HDFStore(fp, 'r') as store:
            for key in store.keys():
                sid = key.strip('/')
                df = store[key]
                # Ensure the index is a DatetimeIndex
                df.index = pd.to_datetime(df.index)
                all_stocks[sid] = df
                logging.debug("Loaded %d rows for stock %s", len(df), sid)
    except Exception as e:
        logging.error(f"Failed to read HDF5 file at {fp}. Error: {e}")
        return {}
    logging.info("Loaded data for %d stocks.", len(all_stocks))
    return all_stocks

# -----------------------------------------------------------------------------
# 7  Build pipeline
# -----------------------------------------------------------------------------

def build(cfg: BuilderConfig, all_stocks: Dict[str, pd.DataFrame]):
    logging.info("Configuration:\n%s", cfg.model_dump_json(indent=2, exclude={"input_hdf5_path"}))
    ind_cols = derive_indicator_cols(cfg)
    feat_names = ["Open_norm_log", "High_norm_log", "Low_norm_log", "Close_norm_log"] + ind_cols
    col_to_idx = {n: i for i, n in enumerate(feat_names)}

    if not all_stocks:
        logging.error("No stock data provided to build function.")
        sys.exit(1)

    if cfg.debug_single_stock:
        logging.warning("--- DEBUG MODE: Processing a single stock ---")
        cfg.num_workers = 1
        first_sid = next(iter(all_stocks))
        all_stocks = {first_sid: all_stocks[first_sid]}

    st_builder = StockBuilder(cfg, ind_cols)
    X_parts, y_parts, d_parts, sid_nested = [], [], [], []

    # Use a with statement for the executor
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as exe:
        futs = {exe.submit(st_builder, df, sid): sid for sid, df in all_stocks.items()}
        prog_desc = "Building (Debug)" if cfg.debug_single_stock else "Building"
        for fut in tqdm(as_completed(futs), total=len(futs), desc=prog_desc, unit="stock"):
            try:
                res = fut.result()
            except Exception as exc:
                logging.error("Worker failed for %s - %s", futs[fut], exc)
                continue
            if res:
                bundle = res[0]
                X_parts.append(bundle["features"])
                y_parts.append(bundle["targets"])
                d_parts.append(bundle["dates"])
                sid_nested.append([bundle["stock"]] * bundle["features"].shape[0])

    if not X_parts:
        logging.error("No data generated - aborting"); sys.exit(1)

    X_all = np.concatenate(X_parts, 0)
    y_all = np.concatenate(y_parts, 0)
    dates_s10 = np.concatenate(d_parts, 0)
    sids = np.concatenate(sid_nested, 0)
    dates_str = [d.decode("utf-8") for d in dates_s10]

    if cfg.data_start_cutoff:
        cut_ts = pd.Timestamp(cfg.data_start_cutoff)
        keep = pd.Series(pd.to_datetime(dates_str)) >= cut_ts
        X_all, y_all, sids = X_all[keep.values], y_all[keep.values], sids[keep.values]
        dates_str = [s for k, s in zip(keep.values, dates_str) if k]
        logging.info("After cutoff %s - remaining %d samples", cut_ts.date(), X_all.shape[0])
        if not dates_str:
            logging.error("No data left after cutoff - abort"); sys.exit(1)

    tgt_ts = pd.to_datetime(dates_str)
    idx_df = pd.DataFrame({"idx": np.arange(len(tgt_ts)), "tgt": tgt_ts}).sort_values("tgt", kind="mergesort")
    last_t = idx_df["tgt"].iat[-1]
    test_start = last_t.normalize() - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    val_start = test_start - pd.DateOffset(years=1)

    # Use target dates for splitting
    test_m = idx_df["tgt"] >= test_start
    val_m = (idx_df["tgt"] >= val_start) & (idx_df["tgt"] < test_start)
    train_m = ~(test_m | val_m)

    ix_tr, ix_val, ix_te = idx_df.loc[train_m, "idx"].to_numpy(int), idx_df.loc[val_m, "idx"].to_numpy(int), idx_df.loc[test_m, "idx"].to_numpy(int)
    logging.info("Split sizes - train:%d  val:%d  test:%d", len(ix_tr), len(ix_val), len(ix_te))
    if not len(ix_tr):
        logging.error("Train split empty - abort"); sys.exit(1)

    X_tr, y_tr = X_all[ix_tr], y_all[ix_tr]
    X_val, y_val = X_all[ix_val], y_all[ix_val]
    X_te, y_te = X_all[ix_te], y_all[ix_te]
    d_tr, d_val, d_te = [dates_str[i] for i in ix_tr], [dates_str[i] for i in ix_val], [dates_str[i] for i in ix_te]
    s_tr, s_val, s_te = sids[ix_tr], sids[ix_val], sids[ix_te]

    tr_min, tr_max = (min(d_tr), max(d_tr)) if d_tr else (None, None)
    w_tr, w_val, w_te = loss_weights(d_tr, tr_min, tr_max), loss_weights(d_val, tr_min, tr_max), loss_weights(d_te, tr_min, tr_max)

    seq_len, feat_dim = X_tr.shape[1], X_tr.shape[2]
    rshape = lambda a: a.reshape(-1, feat_dim)
    X_tr_r, X_val_r, X_te_r = rshape(X_tr), rshape(X_val), rshape(X_te)

    for fname in cfg.features_to_transform_with_asinh:
        if fname in col_to_idx:
            col = col_to_idx[fname]
            logging.info("arcsinh transform on feature %s (col %d)", fname, col)
            for arr in (X_tr_r, X_val_r, X_te_r):
                if arr.size:
                    arr[:, col] = np.arcsinh(arr[:, col])

    scaler: Optional[StandardScaler] = None
    if cfg.standardize_all_features:
        scaler = StandardScaler()
        scaler.fit(X_tr_r)
        X_tr_r = scaler.transform(X_tr_r)
        if X_val_r.size: X_val_r = scaler.transform(X_val_r)
        if X_te_r.size: X_te_r = scaler.transform(X_te_r)
        if cfg.scaler_save_path:
            joblib.dump(scaler, cfg.scaler_save_path)
            logging.info("Scaler saved to %s", cfg.scaler_save_path)

    def back(a_r: np.ndarray, nsamples: int) -> np.ndarray:
        return a_r.reshape(nsamples, seq_len, feat_dim).astype(np.float32)

    X_tr = back(X_tr_r, len(ix_tr))
    X_val = back(X_val_r, len(ix_val)) if X_val_r.size else np.empty((0, seq_len, feat_dim), dtype=np.float32)
    X_te = back(X_te_r, len(ix_te)) if X_te_r.size else np.empty((0, seq_len, feat_dim), dtype=np.float32)

    logging.info("Saving dataset to %s", cfg.output_hdf5_file_with_splits)
    with h5py.File(cfg.output_hdf5_file_with_splits, "w", libver="latest") as hf:
        str_dt = h5py.string_dtype("utf-8")
        hf.create_dataset("feature_names", data=np.array(feat_names, dtype=object), dtype=str_dt)

        def _save(grp_name, X, y, d, s, w):
            grp = hf.create_group(grp_name)
            grp.create_dataset("features", data=X, dtype="float32", compression="gzip")
            grp.create_dataset("targets", data=y, dtype="float32", compression="gzip")
            grp.create_dataset("target_dates", data=[x.encode("utf-8") for x in d], dtype=str_dt, compression="gzip")
            grp.create_dataset("stock_ids", data=s.astype(str_dt), compression="gzip")
            grp.create_dataset("sample_loss_weights", data=w, dtype="float32", compression="gzip")

        _save("train", X_tr, y_tr, d_tr, s_tr, w_tr)
        _save("validation", X_val, y_val, d_val, s_val, w_val)
        _save("test", X_te, y_te, d_te, s_te, w_te)

        if scaler is not None:
            sg = hf.create_group("scaler_info")
            sg.create_dataset("mean", data=scaler.mean_)
            sg.create_dataset("scale", data=scaler.scale_)
            sg.attrs["n_features_in"] = scaler.n_features_in_
            sg.attrs["n_samples_seen"] = int(scaler.n_samples_seen_)

    logging.info("✅ Build complete - %s", cfg.output_hdf5_file_with_splits)


# -----------------------------------------------------------------------------
# 8  Main execution
# -----------------------------------------------------------------------------

def run_dataset_build():
    """
    Configures and runs the full data processing pipeline.
    """
    # Note: freeze_support() is needed for multiprocessing to work correctly when
    # the script is frozen into an executable.
    mp.freeze_support()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", force=True)

    # Hardcoded configuration based on user request and original script defaults.
    # IMPORTANT: The user must ensure these paths are correct for their system.
    config_data = {
        "input_hdf5_path": r"C:\Users\robca\OneDrive\Documents\Experiment\UK analysis 3\IBAPI data\lse_daily_15y.h5",
        "output_hdf5_file_with_splits": r"C:\Users\robca\OneDrive\Documents\Experiment\UK analysis 3\IBAPI data\IBAPI_dataset.h5",
        "window_size": 14,
        "min_rows": 60,
        "num_workers": max(mp.cpu_count() - 1, 1), # Leave one core free
        "data_start_cutoff": "2010-01-01",
        "strict_window_split": False, # Changed to False to match original script's effective behavior
        "debug_checks": False,
        "debug_single_stock": False,
        "features_to_transform_with_asinh": [],
        "standardize_all_features": True,
        "indicators": [
            {"method": "rsi", "params": {"length": 14}, "output_cols": ["RSI_14"]},
            {"method": "adx", "params": {"length": 14}, "output_cols": ["ADX_14", "DMP_14"]},
            {"method": "bbands", "params": {"length": 20, "std": 2}, "output_cols": ["BBB_20_2.0"]},
            {"method": "stoch", "params": {"k": 14, "d": 3, "smooth_k": 3}, "output_cols": ["STOCHk_14_3_3", "STOCHd_14_3_3"]},
            {"method": "roc", "params": {"length": 12}, "output_cols": ["ROC_12"]},
            {"method": "willr", "params": {"length": 14}, "output_cols": ["WILLR_14"]},
        ]
    }

    try:
        cfg = BuilderConfig.model_validate(config_data)
    except ValidationError as exc:
        logging.critical("Configuration validation error:\n%s", exc)
        sys.exit(1)

    # 1. Load the raw data
    all_stocks = load_data_from_hdf5(cfg.input_hdf5_path)
    if not all_stocks:
        logging.error("No data was loaded from the input file. Please check the path and file integrity. Exiting.")
        sys.exit(1)

    # 2. Build the dataset
    try:
        build(cfg, all_stocks)
    except Exception as exc:
        logging.critical("An unhandled error occurred during the build process: %s", exc, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", force=True)
    run_dataset_build()