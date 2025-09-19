# validate_tickers.py
import threading
import time
import re
import sys
from pathlib import Path

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

import config

class TickerValidator(EWrapper, EClient):
    """
    Connects to IBKR and validates a list of tickers to ensure they
    can be resolved to a single, unique contract.
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.valid_tickers = []
        self.failed_tickers = {}
        self.current_ticker = None
        self.contract_details_received = threading.Event()
        self.connection_acknowledged_event = threading.Event()
        self.request_id_counter = 0

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        print("API connection successful. Ready to validate tickers.")
        self.connection_acknowledged_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="", errorTime=0):
        if errorCode in [162, 2104, 2106, 2107, 2158, 2176]:
            return
        
        if reqId != -1:
            self.failed_tickers[self.current_ticker] = f"Error Code {errorCode}: {errorString}"
            self.contract_details_received.set()

    def contractDetails(self, reqId, contractDetails):
        self.valid_tickers.append(contractDetails.contract.symbol)
        self.contract_details_received.set()

    def contractDetailsEnd(self, reqId):
        self.contract_details_received.set()

    def create_lse_stock_contract(self, symbol: str) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "LSE"
        contract.currency = "GBP"
        return contract

    def run_validation(self, tickers: list):
        print("Starting ticker validation process...")
        for ticker in tickers:
            self.current_ticker = ticker
            self.contract_details_received.clear()
            
            contract = self.create_lse_stock_contract(ticker)
            self.reqContractDetails(self.request_id_counter, contract)
            self.request_id_counter += 1

            # Wait for a response, but with a timeout
            if not self.contract_details_received.wait(5):
                self.failed_tickers[ticker] = "Timeout: No response from API."
            
            time.sleep(0.2) # Pacing to respect API limits

        print("\nValidation complete.")


def get_and_sanitize_tickers(file_path: str) -> list:
    """Reads and sanitizes tickers from a text file."""
    tickers = []
    with open(file_path, 'r') as f:
        for line in f:
            t = re.sub(r'[^a-zA-Z0-9]', '', line.strip())
            if t:
                tickers.append(t)
    
    # Remove duplicates
    return sorted(list(set(tickers)))


if __name__ == "__main__":
    app = TickerValidator()
    app.connect(config.IB_HOST, config.IB_PORT, clientId=999) # Use a different client ID

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    if not app.connection_acknowledged_event.wait(10):
        print("Failed to connect to TWS/Gateway.")
        sys.exit(1)

    try:
        raw_tickers = get_and_sanitize_tickers(str(config.TICKER_FILE))
        app.run_validation(raw_tickers)
        
        # Write the results to files
        valid_tickers_path = config.DATA_DIR / "valid_tickers.txt"
        with open(valid_tickers_path, 'w') as f:
            for ticker in sorted(app.valid_tickers):
                f.write(f"{ticker}\n")
        
        failed_tickers_path = config.DATA_DIR / "failed_tickers.log"
        with open(failed_tickers_path, 'w') as f:
            for ticker, reason in app.failed_tickers.items():
                f.write(f"{ticker}: {reason}\n")

        print(f"\nSuccessfully validated {len(app.valid_tickers)} tickers.")
        print(f"Results saved to: {valid_tickers_path}")
        print(f"Failed tickers logged in: {failed_tickers_path}")

    finally:
        app.disconnect()