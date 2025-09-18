# initialise_dataset.py
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

import config

class IBKRDataDownloader(EWrapper, EClient):
    """
    Downloads historical data from Interactive Brokers API,
    respecting pacing limitations and handling errors robustly.
    """

    def __init__(self):
        EClient.__init__(self, self)
        self.data_dict = {}
        self.failed_tickers = [] # List to store tickers that failed
        self.current_ticker = None
        self.data_received_event = threading.Event()
        self.connection_acknowledged_event = threading.Event()

    def nextValidId(self, orderId: int):
        """Confirms that the connection is established and ready."""
        super().nextValidId(orderId)
        print("API connection successful. Ready to send requests.")
        self.connection_acknowledged_event.set()

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = "", errorTime: int = 0):
        """Catches errors from the API."""
        if errorCode in [162, 2104, 2106, 2107, 2158, 2176]:
            print(f"INFO (ReqId: {reqId}): {errorString}")
            return

        print(f"ERROR - ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")
        
        if reqId != -1 and self.data_received_event:
            self.data_received_event.set()

    def historicalData(self, reqId: int, bar):
        """Callback that receives each historical data bar."""
        data = {
            "Date": bar.date, "Open": bar.open, "High": bar.high,
            "Low": bar.low, "Close": bar.close, "Volume": bar.volume,
        }
        if self.current_ticker not in self.data_dict:
            self.data_dict[self.current_ticker] = []
        self.data_dict[self.current_ticker].append(data)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Callback triggered when all historical data has been received."""
        super().historicalDataEnd(reqId, start, end)
        print(f"Finished receiving data for request {reqId}.")
        self.data_received_event.set()

    def create_lse_stock_contract(self, symbol: str) -> Contract:
        """Creates an IB Contract object for a stock listed on the LSE."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "LSE"
        contract.currency = "GBP"
        return contract

    def run_download(self, tickers: list, duration: str, bar_size: str = "1 day"):
        """Main method to loop through tickers and download data."""
        print("Starting data download process...")
        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            self.current_ticker = ticker
            self.data_received_event.clear()

            print(f"\n[{i+1}/{total_tickers}] Requesting data for: {ticker}")
            contract = self.create_lse_stock_contract(ticker)
            reqId = i

            self.reqHistoricalData(
                reqId, contract, "", duration, bar_size, "TRADES", 1, 1, False, []
            )

            timeout = 30
            event_is_set = self.data_received_event.wait(timeout)

            if not event_is_set:
                print(f"TIMEOUT waiting for data for {ticker}. Cancelling request.")
                self.cancelHistoricalData(reqId)
                self.failed_tickers.append(ticker)
                if ticker in self.data_dict and isinstance(self.data_dict[ticker], list):
                    del self.data_dict[ticker]
            else:
                if ticker in self.data_dict and self.data_dict[ticker]:
                    df = pd.DataFrame(self.data_dict[ticker])
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                    df.set_index('Date', inplace=True)
                    df['Ticker'] = ticker
                    self.data_dict[ticker] = df
                    print(f"SUCCESS: Stored {len(df)} days of data for {ticker}.")
                else:
                    print(f"FAILURE: No data received for {ticker}.")
                    self.failed_tickers.append(ticker)
                    if ticker in self.data_dict:
                        del self.data_dict[ticker]

            # Respect API pacing rules
            print("Pausing for 11 seconds to respect API limits...")
            time.sleep(11)

        print("\nAll tickers processed.")


def get_lse_tickers_from_txt(file_path: str) -> list:
    """Reads tickers from a plain text file, one ticker per line."""
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"Found {len(tickers)} tickers in '{file_path}'.")
        return tickers
    except FileNotFoundError:
        print(f"Error: Ticker file not found at the specified path: {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the text file: {e}")
        return []

def sanitize_key(key: str) -> str:
    """Sanitizes a string to be a valid HDF5 key."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', key)

def run_full_download(tickers: list, duration: str, output_path: str):
    """
    Connects to IB, downloads historical data for the given tickers,
    and saves the result to an HDF5 file.
    """
    app = IBKRDataDownloader()
    app.connect(config.IB_HOST, config.IB_PORT, clientId=config.IB_CLIENT_ID)
    print(f"Connecting to TWS on {config.IB_HOST}:{config.IB_PORT}...")

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    connection_confirmed = app.connection_acknowledged_event.wait(timeout=10)
    if not connection_confirmed:
        print("Failed to connect to TWS/Gateway.")
        app.disconnect()
        return

    try:
        app.run_download(tickers, duration=duration)

        print(f"\nSaving all collected data to '{output_path}'...")
        valid_data = {k: v for k, v in app.data_dict.items() if isinstance(v, pd.DataFrame) and not v.empty}

        if not valid_data:
            print("No valid data was downloaded. HDF5 file will not be created.")
        else:
            with pd.HDFStore(output_path, 'w') as store:
                for ticker, df in valid_data.items():
                    safe_key = sanitize_key(ticker)
                    store.put(safe_key, df, format='table', data_columns=True)
            print(f"Successfully saved data for {len(valid_data)} tickers.")

    finally:
        if app.failed_tickers:
            print("\n-------------------------------------------")
            print("The following tickers failed to download:")
            for ticker in app.failed_tickers:
                print(f"- {ticker}")
            print("-------------------------------------------")
        else:
            print("\nAll tickers were processed successfully.")

        print("Disconnecting from API...")
        app.disconnect()


if __name__ == "__main__":
    ticker_file = str(config.TICKER_FILE)
    lse_tickers = get_lse_tickers_from_txt(ticker_file)
    if lse_tickers:
        run_full_download(
            tickers=lse_tickers,
            duration=f"{config.RAW_DATA_HISTORY_DAYS} D",
            output_path=str(config.RAW_DATABASE_PATH)
        )
