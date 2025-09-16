# initialise_dataset.py
import logging
import sys
from pathlib import Path

# Import the core logic from your existing, trusted scripts
import IBAPI_data_extraction
import rawtodatasetIBAPI
import config

def main():
    """
    Orchestrates the one-time process of initializing the bot's datasets.
    
    Step 1: Download a fresh set of raw historical data from IBKR.
    Step 2: Process that raw data to create the model-ready feature dataset and scaler.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    logger = logging.getLogger("Initializer")

    logger.info("--- Starting One-Time Dataset Initialization ---")

    # --- Step 1: Download fresh raw data ---
    logger.info("STEP 1: Downloading fresh raw historical data from Interactive Brokers...")
    try:
        ticker_file_path = Path(config.TICKER_FILE)
        tickers = IBAPI_data_extraction.get_lse_tickers_from_txt(str(ticker_file_path))
        if not tickers:
            logger.critical("Ticker file is empty. Aborting.")
            return

        # We download 2 months of data. This is fast and more than enough
        # to calculate a 20-period indicator for a 14-day window.
        IBAPI_data_extraction.run_full_download(
            tickers=tickers,
            duration="2 M", # Fetching 2 months is sufficient and fast
            output_path=config.RAW_DATABASE_PATH
        )
        logger.info("STEP 1 COMPLETE: Raw data download finished.")

    except Exception as e:
        logger.critical(f"A critical error occurred during data download: {e}", exc_info=True)
        return

    # --- Step 2: Process raw data into model-ready format ---
    logger.info("\n--- STEP 2: Processing raw data to create model-ready dataset and scaler ---")
    try:
        rawtodatasetIBAPI.run_dataset_build()
        logger.info("STEP 2 COMPLETE: Model-ready dataset and scaler have been created.")
    except Exception as e:
        logger.critical(f"A critical error occurred during dataset processing: {e}", exc_info=True)
        return
        
    logger.info("\n✅ Initialization successful! The bot is ready for its first scheduled run.")

if __name__ == "__main__":
    main()