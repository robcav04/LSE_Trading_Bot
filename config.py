# config.py
from pathlib import Path

# --- FILE PATHS 📂 ---
# This section defines where all the necessary files are located.
# The bot will fail at startup if these paths are incorrect.

# The root directory of your project
BASE_DIR = Path(__file__).resolve().parent

# Data sub-directory
DATA_DIR = BASE_DIR / "data"
TICKER_FILE = DATA_DIR / "valid_tickers.txt"

# Path to the RAW historical data file (read by ModelHandler, written by overnight/seeder)
RAW_DATABASE_PATH = DATA_DIR / "lse_daily_15y.h5"

# Path to the PROCESSED, model-ready data file (created by the seeder)
PROCESSED_DATABASE_PATH = DATA_DIR / "IBAPI_dataset.h5"

# Path to the fitted scaler object (created by the seeder, read by ModelHandler)
SCALER_PATH = DATA_DIR / "IBAPI_dataset_scaler.joblib"

# Model sub-directory
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model_torchscript.pt"


# --- IB GATEWAY SETTINGS 🔌 ---
# These must match your IB Gateway configuration exactly.
IB_HOST = "127.0.0.1"  # Keep this as is for a local connection
IB_PORT = 4002         # Default for Paper Trading with IB Gateway
IB_CLIENT_ID = 101     # A unique ID for this bot instance to avoid conflicts


# --- STRATEGY PARAMETERS 📈 ---
# These variables control the bot's trading behavior.
RISK_FACTOR = 0.5 # Use 95% of available funds for a trade
RAW_DATA_HISTORY_DAYS = 100
MARKET_CALENDAR = "LSE"
TIMEZONE = "Europe/London"

# How many minutes before market close to submit the Market-on-Close order
MOC_SUBMISSION_WINDOW_MINUTES = 15

# How many seconds to wait for opening prices after the market opens before giving up on a stock
OPENING_PRICE_TIMEOUT_SECONDS = 90

# --- ALGO PARAMETERS ---
# Target percentage of the average closing auction volume for the 'ClosePrice' Algo
# A value of 0.1 means the algo will target 10% of the closing volume.
CLOSE_ALGO_TARGET_PERCENT_OF_VOLUME = 0.1

# --- MODEL & FEATURE SPECIFICATIONS 🔬 ---
# These settings must perfectly match the configuration used to train your model.
# Any mismatch here will lead to incorrect predictions.

WINDOW_SIZE = 14

# The exact order of features your model expects.
# This is derived from your `rawtodatasetIBAPI.py` script.
FEATURE_NAMES = [
    "Open_norm_log", "High_norm_log", "Low_norm_log", "Close_norm_log",
    "RSI_14", "ADX_14", "DMP_14", "BBB_20_2.0", "STOCHk_14_3_3",
    "STOCHd_14_3_3", "ROC_12", "WILLR_14"
]

# The exact indicator configuration used during training.
# This is copied directly from your `rawtodatasetIBAPI.py` script.
INDICATORS = [
    {"method": "rsi", "params": {"length": 14}, "output_cols": ["RSI_14"]},
    {"method": "adx", "params": {"length": 14}, "output_cols": ["ADX_14", "DMP_14"]},
    {"method": "bbands", "params": {"length": 20, "std": 2}, "output_cols": ["BBB_20_2.0"]},
    {"method": "stoch", "params": {"k": 14, "d": 3, "smooth_k": 3}, "output_cols": ["STOCHk_14_3_3", "STOCHd_14_3_3"]},
    {"method": "roc", "params": {"length": 12}, "output_cols": ["ROC_12"]},
    {"method": "willr", "params": {"length": 14}, "output_cols": ["WILLR_14"]},
]
