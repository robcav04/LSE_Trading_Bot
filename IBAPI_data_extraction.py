import re
import threading
import time
from datetime import datetime

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
        # Use threading.Event for synchronization
        self.data_received_event = threading.Event()
        self.connection_acknowledged_event = threading.Event()

    def nextValidId(self, orderId: int):
        """Confirms that the connection is established and ready."""
        super().nextValidId(orderId)
        print("API connection successful. Ready to send requests.")
        self.connection_acknowledged_event.set()

    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """Catches errors from the API."""
        # --- MODIFIED SECTION ---
        # These are informational messages and should not be treated as errors
        # that terminate a data request.
        # 2104/2106/2158 are market data status messages.
        # 2176 is a warning about fractional share size rules that does not affect the download.
        if errorCode in [162, 2104, 2106, 2158, 2176]:
            print(f"INFO: {errorString}")
            return # Do not proceed further for these non-critical messages

        print(f"ERROR - ReqId: {reqId}, Code: {errorCode}, Message: {errorString}")
        # For genuine errors, signal that the request is complete to unblock the main thread
        if reqId != -1:
            self.data_received_event.set()

    def historicalData(self, reqId: int, bar):
        """Callback that receives each historical data bar."""
        data = {
            "Date": bar.date,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume,
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

    def run_full_download(tickers: list, duration: str, output_path: str):
        """
        Connects to IB, downloads historical data for the given tickers,
        and saves the result to an HDF5 file.
        """
        TWS_HOST = config.IB_HOST
        TWS_PORT = config.IB_PORT
        CLIENT_ID = config.IB_CLIENT_ID

        app = IBKRDataDownloader()
        app.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
        print(f"Connecting to TWS on {TWS_HOST}:{TWS_PORT}...")

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
    TWS_HOST = config.IB_HOST
    TWS_PORT = config.IB_PORT
    CLIENT_ID = config.IB_CLIENT_ID

    app = IBKRDataDownloader()
    app.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
    print(f"Connecting to TWS on {TWS_HOST}:{TWS_PORT}...")

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
    # This block allows running the script standalone for a full 15-year download
    ticker_file = r"C:\Users\robca\OneDrive\Documents\Experiment\UK analysis 3\IBAPI data\TIcker_list.txt"
    lse_tickers = get_lse_tickers_from_txt(ticker_file)
    if lse_tickers:
        run_full_download(
            tickers=lse_tickers,
            duration="15 Y",
            output_path=r"C:\Users\robca\OneDrive\Documents\Experiment\UK analysis 3\IBAPI data\lse_daily_15y.h5"
        )