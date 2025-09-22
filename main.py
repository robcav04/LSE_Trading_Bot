# main.py
import threading
import time
import argparse
import logging
import signal
import sys
import math
from datetime import datetime
from pathlib import Path
import queue
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, BarData
from ibapi.contract import ContractDetails

import config
from utils import setup_logging, get_lse_calendar, get_market_schedule_for_date
from data_manager import DataManager
from model_handler import ModelHandler

# --- Global Shutdown Event ---
SHUTDOWN_EVENT = threading.Event()


# -----------------------------
# Non-IB helpers & validations
# -----------------------------

def _validate_config_and_env() -> None:
    """Early, explicit config and environment validation for safer failures."""
    required_attrs = [
        "IB_HOST", "IB_PORT", "IB_CLIENT_ID",
        "TIMEZONE", "MODEL_PATH", "SCALER_PATH",
        "BASE_DIR", "TICKER_FILE",
        "MOC_SUBMISSION_WINDOW_MINUTES", "OPENING_PRICE_TIMEOUT_SECONDS",
    ]
    missing = [a for a in required_attrs if not hasattr(config, a)]
    if missing:
        raise ValueError(f"Missing required config attributes: {', '.join(missing)}")

    if not isinstance(config.IB_PORT, int) or config.IB_PORT <= 0:
        raise ValueError("config.IB_PORT must be a positive integer.")
    if not isinstance(config.IB_CLIENT_ID, int) or config.IB_CLIENT_ID < 0:
        raise ValueError("config.IB_CLIENT_ID must be a non-negative integer.")
    if not isinstance(config.MOC_SUBMISSION_WINDOW_MINUTES, (int, float)) or config.MOC_SUBMISSION_WINDOW_MINUTES <= 0:
        raise ValueError("config.MOC_SUBMISSION_WINDOW_MINUTES must be a positive number of minutes.")
    if not isinstance(config.OPENING_PRICE_TIMEOUT_SECONDS, (int, float)) or config.OPENING_PRICE_TIMEOUT_SECONDS <= 0:
        raise ValueError("config.OPENING_PRICE_TIMEOUT_SECONDS must be a positive number of seconds.")

    # Timezone validation
    try:
        pytz.timezone(config.TIMEZONE)
    except Exception as e:
        raise ValueError(f"Invalid timezone '{config.TIMEZONE}': {e}")

    # Paths & directories
    model_path = Path(config.MODEL_PATH)
    scaler_path = Path(config.SCALER_PATH)
    tickers_path = Path(config.TICKER_FILE)
    logs_dir = Path(config.BASE_DIR) / "logs"

    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"SCALER_PATH does not exist: {scaler_path}")
    if not tickers_path.exists():
        raise FileNotFoundError(f"TICKER_FILE does not exist: {tickers_path}")

    logs_dir.mkdir(parents=True, exist_ok=True)


def _load_tickers(file_path: Path) -> List[str]:
    """Load, clean, and dedupe tickers from a file."""
    tickers: List[str] = []
    with file_path.open("r") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            tickers.append(t)

    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    if not deduped:
        raise ValueError(f"No valid tickers found in {file_path}.")
    return deduped


def _safe_sleep_until(target_dt: datetime, tz: str) -> None:
    """Sleep in short intervals until target_dt, respecting SHUTDOWN_EVENT."""
    tzinfo = pytz.timezone(tz)
    while not SHUTDOWN_EVENT.is_set():
        now = datetime.now(tzinfo)
        remaining = (target_dt - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))


def _validate_schedule(schedule: Optional[Dict[str, pd.Timestamp]]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Validate schedule dict ensuring 'open'/'close' exist and are timestamps."""
    if schedule is None:
        raise ValueError("No market schedule available for today (market closed or calendar error).")
    open_ts = schedule.get("open")
    close_ts = schedule.get("close")
    if open_ts is None or close_ts is None or pd.isna(open_ts) or pd.isna(close_ts):
        raise ValueError("Schedule missing 'open' or 'close' time.")
    if not isinstance(open_ts, pd.Timestamp) or not isinstance(close_ts, pd.Timestamp):
        raise TypeError("Schedule 'open'/'close' must be pandas.Timestamp.")
    if close_ts <= open_ts:
        raise ValueError(f"Schedule invalid: close ({close_ts}) is not after open ({open_ts}).")
    return open_ts, close_ts


def _is_finite_positive(x) -> bool:
    """Checks if a value is a finite, positive number."""
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


class IBApp(EWrapper, EClient):
    """
    The main trading bot application class.
    Handles all communication with IB Gateway and orchestrates the trading logic.
    """
    REQ_ID_ACCOUNT_SUMMARY = 9001  # Class constant for clarity

    def __init__(self, tickers: list):
        EClient.__init__(self, self)
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("IBApp requires a non-empty list of tickers.")
        self.all_tickers = tickers
        self.tradable_tickers: List[str] = []
        self.logger = logging.getLogger("TradingBot")

        # --- State Management ---
        self._id_lock = threading.Lock()
        self.next_order_id: Optional[int] = None
        self.current_positions: Dict[str, dict] = {}
        self.open_orders: Dict[int, dict] = {}
        self.permId_to_orderId: Dict[int, int] = {}
        self.net_liquidation_value: float = 0.0
        self.what_if_result: Optional[dict] = None

        # --- Data & Model Handlers ---
        self.data_manager = DataManager()
        self.model_handler = ModelHandler(config.MODEL_PATH, config.SCALER_PATH)
        self.historical_data_cache: Dict[str, pd.DataFrame] = {}

        # --- Live Data Collection ---
        self.open_prices: Dict[str, float] = {}
        self.open_price_req_ids: Dict[int, str] = {}
        self.open_price_ticker_map: Dict[str, int] = {}

        # --- Asynchronous Request Management ---
        self.historical_data_queue: queue.Queue = queue.Queue()
        self.active_requests: Dict[int, threading.Event] = {}
        self.active_contract_requests: Dict[int, dict] = {}

        # --- Threading Events for Synchronization ---
        self.connection_event = threading.Event()
        self.next_valid_id_event = threading.Event()
        self.position_end_event = threading.Event()
        self.open_order_end_event = threading.Event()
        self.account_summary_event = threading.Event()

    # --- EWrapper Callbacks (Handles incoming messages from IB) ---
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = "", errorTime: int = 0):
        """Catches errors from the API."""
        # Informational messages
        if errorCode in [162, 2104, 2106, 2107, 2158, 2176]:
            self.logger.info(f"INFO (ReqId: {reqId}): {errorString}")
            if reqId in self.active_contract_requests:
                self.active_contract_requests[reqId]['event'].set() # Let valid info messages also unblock
            return

        self.logger.error(f"ERROR - ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}")
        
        # Handle failed contract detail requests
        if reqId in self.active_contract_requests:
            ticker = self.active_contract_requests[reqId]['ticker']
            if errorCode == 200: # "No security definition has been found for the request"
                self.logger.warning(f"Could not find contract for {ticker}. It will be excluded.")
            else:
                 self.logger.error(f"API error for {ticker} contract details.")
            self.active_contract_requests[reqId]['event'].set()
        
        # Handle historical data request timeouts or failures
        if reqId in self.active_requests:
            self.active_requests[reqId].set()


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_order_id = orderId
        self.next_valid_id_event.set()
        self.logger.info(f"Received next valid order ID: {orderId}")

    def connectAck(self):
        self.logger.info("Connection Acknowledged.")
        self.connection_event.set()

    def connectionClosed(self):
        self.logger.error("Connection closed.")
        self.connection_event.clear() # Clear event on close

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        if position != 0:
            self.current_positions[contract.symbol] = {'pos': position, 'avgCost': avgCost, 'conId': contract.conId}

    def positionEnd(self):
        self.position_end_event.set()

    def openOrder(self, orderId, contract, order, orderState):
        self.open_orders[orderId] = {'contract': contract.symbol, 'status': orderState.status}
        if getattr(order, "whatIf", False):
            self.logger.info("Received What-If order state.")
            self.what_if_result = {
                'initMargin': orderState.initMarginChange,
                'maintMargin': orderState.maintMarginChange,
                'equityWithLoan': orderState.equityWithLoanChange,
            }
        self.permId_to_orderId[order.permId] = orderId

    def openOrderEnd(self):
        self.open_order_end_event.set()

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.logger.info(f"Order Status: {orderId}, Status: {status}, Filled: {filled}, Remaining: {remaining}")
        if orderId in self.open_orders:
            self.open_orders[orderId]['status'] = status
        if status in ['Filled', 'Cancelled', 'ApiCancelled']:
            if orderId in self.open_orders:
                del self.open_orders[orderId]

    def execDetails(self, reqId, contract, execution):
        self.logger.info(f"Execution: {execution.orderId}, {contract.symbol}, {execution.side} {execution.shares} @ {execution.price}")

    def accountSummary(self, reqId, account, tag, value, currency):
        if tag == "NetLiquidation":
            try:
                self.net_liquidation_value = float(value)
            except (ValueError, TypeError):
                self.logger.error(f"Failed to parse NetLiquidation value '{value}'")
                self.net_liquidation_value = 0.0
            self.account_summary_event.set()

    def accountSummaryEnd(self, reqId: int):
        self.cancelAccountSummary(reqId)

    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        # Only use tick type 14 (OPEN_TICK) for the opening price.
        if tickType == 14 and reqId in self.open_price_req_ids:
            ticker = self.open_price_req_ids[reqId]
            if ticker not in self.open_prices and _is_finite_positive(price):
                self.open_prices[ticker] = price
                self.logger.info(f"Received opening price for {ticker} (TickType 14): {price}")
                self.cancelMktData(reqId)

    def historicalData(self, reqId: int, bar: BarData):
        data = {
            "Date": bar.date, "Open": bar.open, "High": bar.high,
            "Low": bar.low, "Close": bar.close, "Volume": bar.volume,
        }
        self.historical_data_queue.put({'reqId': reqId, 'data': data})

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        if reqId in self.active_requests:
            self.active_requests[reqId].set()
            self.logger.info(f"Historical data end for ReqId: {reqId}")

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        """Receives contract details and confirms a ticker is valid."""
        if reqId in self.active_contract_requests:
            ticker = self.active_contract_requests[reqId]['ticker']
            self.logger.info(f"Successfully verified contract for {ticker}.")
            self.tradable_tickers.append(ticker)
            self.active_contract_requests[reqId]['event'].set()
    
    def contractDetailsEnd(self, reqId: int):
        """Signals the end of a contract details request."""
        if reqId in self.active_contract_requests:
            # If we get an end message but no details, it means no contract was found.
            # The error callback usually handles this, but this is a good fallback.
            self.active_contract_requests[reqId]['event'].set()


    # --- Core Bot Methods ---

    def _create_contract(self, symbol: str) -> Contract:
        """Creates a Contract object for a LSE stock."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "LSE"
        contract.currency = "GBP"
        contract.primaryExchange = "LSE"
        return contract

    def _create_order(self, action: str, quantity: float, order_type: str = "MKT") -> Order:
        """Creates a basic Order object."""
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        order.tif = "GTC" if order_type == "MOC" else "DAY"
        return order

    def connect_and_start(self):
        """Connects to IB Gateway and starts the message processing thread."""
        try:
            self.connect(config.IB_HOST, config.IB_PORT, clientId=config.IB_CLIENT_ID)
        except Exception as e:
            raise ConnectionError(f"Failed to initiate connection to IB Gateway: {e}")
        self.logger.info(f"Connecting to IB Gateway on {config.IB_HOST}:{config.IB_PORT}...")

        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()

        if not self.connection_event.wait(10):
            raise ConnectionError("Failed to connect to IB Gateway (no connectAck).")

        if not self.next_valid_id_event.wait(5):
            raise TimeoutError("Did not receive nextValidId from IB Gateway.")

    def sync_state(self):
        """Synchronizes bot's state with the broker."""
        self.logger.info("Syncing state with broker...")

        self.position_end_event.clear()
        self.reqPositions()
        if not self.position_end_event.wait(10):
            self.logger.warning("Timeout waiting for position data.")
        self.logger.info(f"Found {len(self.current_positions)} positions: {list(self.current_positions.keys())}")

        self.open_order_end_event.clear()
        self.reqAllOpenOrders()
        if not self.open_order_end_event.wait(10):
            self.logger.warning("Timeout waiting for open order data.")
        self.logger.info(f"Found {len(self.open_orders)} open orders.")

        self.account_summary_event.clear()
        self.reqAccountSummary(self.REQ_ID_ACCOUNT_SUMMARY, "All", "NetLiquidation")
        if not self.account_summary_event.wait(10):
            raise TimeoutError("Timeout waiting for account summary.")
        if not _is_finite_positive(self.net_liquidation_value):
            raise ValueError(f"Net Liquidation Value is invalid: {self.net_liquidation_value}")
        self.logger.info(f"Net Liquidation Value: {self.net_liquidation_value}")

    def prepare_tradable_tickers(self):
        """
        Performs pre-market checks to identify tickers that are accessible
        and have sufficient historical data.
        """
        self.logger.info("--- Starting Pre-Market Ticker Preparation ---")
        
        # 1. Load all available historical data from the local database
        try:
            self.historical_data_cache = self.data_manager.get_all_latest_data()
            if not self.historical_data_cache:
                self.logger.critical("Could not load any historical data. Aborting trading session.")
                return
        except Exception as e:
            self.logger.critical(f"Failed loading historical data during pre-market prep: {e}", exc_info=True)
            return

        # 2. Filter tickers based on data availability and a minimum length of 35 days
        self.logger.info(f"Found historical data for {len(self.historical_data_cache)} tickers.")
        tickers_with_enough_data = [
            ticker for ticker, df in self.historical_data_cache.items() if len(df) >= 35
        ]
        self.logger.info(f"{len(tickers_with_enough_data)} tickers have sufficient historical data.")
        
        if not tickers_with_enough_data:
            self.logger.warning("No tickers have enough data for trading.")
            return

        # 3. Verify contract accessibility with IBKR sequentially
        self.logger.info("Verifying ticker accessibility with Interactive Brokers...")
        self.tradable_tickers = [] # Reset the list for the new session
        req_id_counter = 5000  # High number to avoid collisions with other requests

        for ticker in tickers_with_enough_data:
            if SHUTDOWN_EVENT.is_set():
                self.logger.info("Shutdown requested, aborting ticker verification.")
                break
            
            reqId = req_id_counter
            req_id_counter += 1
            
            event = threading.Event()
            self.active_contract_requests[reqId] = {'ticker': ticker, 'event': event}
            
            contract = self._create_contract(ticker)
            self.logger.debug(f"Requesting contract details for {ticker} with ReqId {reqId}.")
            self.reqContractDetails(reqId, contract)

            event_was_set = event.wait(timeout=10)
            if not event_was_set:
                self.logger.error(f"Timeout waiting for contract details for {ticker}. It will be excluded.")
            
            del self.active_contract_requests[reqId]
            time.sleep(0.2) # Pacing to be safe

        self.logger.info(f"--- Pre-Market Ticker Preparation Complete ---")
        self.logger.info(f"Found {len(self.tradable_tickers)} tradable tickers: {self.tradable_tickers}")


    def run_overnight_session(self):
        """Fetches the previous day's data and updates the local database."""
        self.logger.info("Starting OVERNIGHT data update session.")
        bars_to_update: Dict[str, dict] = {}
        req_id_offset = 1000

        for i, ticker in enumerate(self.all_tickers):
            if SHUTDOWN_EVENT.is_set():
                self.logger.info("Shutdown requested; aborting overnight session loop.")
                break

            reqId = i + req_id_offset
            contract = self._create_contract(ticker)
            completion_event = threading.Event()
            self.active_requests[reqId] = completion_event

            try:
                self.reqHistoricalData(reqId, contract, "", "1 D", "1 day", "TRADES", 1, 1, False, [])
            except Exception as e:
                self.logger.error(f"reqHistoricalData failed for {ticker}: {e}")
                del self.active_requests[reqId]
                continue

            if not completion_event.wait(15):
                self.logger.error(f"Timeout getting historical data for {ticker}. Cancelling.")
                self.cancelHistoricalData(reqId)

            while not self.historical_data_queue.empty():
                try:
                    item = self.historical_data_queue.get_nowait()
                    if item.get('reqId') == reqId:
                        bars_to_update[ticker] = item['data']
                        self.logger.info(f"Fetched daily bar for {ticker}")
                        break
                except queue.Empty:
                    break
            
            del self.active_requests[reqId]
            # PACING: Wait 11 seconds between historical data requests to stay well clear of the 
            # 60 requests per 10 minutes limit.
            time.sleep(11)

        if bars_to_update:
            self.logger.info(f"Finished fetching data. Updating database for {len(bars_to_update)} tickers.")
            self.data_manager.update_raw_database(bars_to_update)
        else:
            self.logger.info("No new bars were fetched to update.")

        self.logger.info("Overnight session complete.")

    def run_trading_session(self):
        """Main logic for a live trading day using pre-vetted tickers."""
        if SHUTDOWN_EVENT.is_set():
            return

        if not self.tradable_tickers or not self.historical_data_cache:
            self.logger.critical("No tradable tickers or historical data available. Aborting.")
            return

        self.request_opening_prices()

        try:
            predictions = self.model_handler.predict(self.historical_data_cache, self.open_prices)
        except Exception as e:
            self.logger.critical(f"Model prediction failed: {e}", exc_info=True)
            return

        if not predictions:
            self.logger.warning("No predictions were generated by the model.")
            return

        valid_predictions = {
            k: v for k, v in predictions.items() 
            if k in self.open_prices and isinstance(v, (int, float)) and math.isfinite(v)
        }
        if not valid_predictions:
            self.logger.warning("No valid predictions with available opening prices.")
            return

        best_ticker = max(valid_predictions, key=lambda k: abs(valid_predictions[k]))
        predicted_log_return = float(valid_predictions[best_ticker])
        action = "BUY" if predicted_log_return > 0 else "SELL"
        self.logger.info(f"Strategy decided: {action} {best_ticker} (Predicted Log Return: {predicted_log_return:.4f})")

        if not _is_finite_positive(self.net_liquidation_value):
            self.logger.error("Cannot trade due to invalid Net Liquidation Value.")
            return
        
        open_price = self.open_prices.get(best_ticker)
        if not _is_finite_positive(open_price):
            self.logger.error(f"Cannot trade due to missing/invalid opening price for {best_ticker}.")
            return
        
        quantity = int(self.net_liquidation_value / open_price)

        if quantity < 1:
            self.logger.error(f"Calculated order size is less than 1 share for {best_ticker}. Aborting.")
            return
        self.logger.info(f"Calculated order size: {quantity} shares.")

        opening_contract = self._create_contract(best_ticker)
        opening_order = self._create_order(action, quantity)

        if not self.execute_what_if_check(opening_contract, opening_order):
            self.logger.critical("What-If check failed. Aborting trade.")
            return

        self.logger.info(f"Placing LIVE OPENING order for {quantity} {best_ticker}...")
        live_order_id = self.get_next_order_id()
        self.placeOrder(live_order_id, opening_contract, opening_order)

        if SHUTDOWN_EVENT.is_set():
            return
        self.monitor_and_close_loop(best_ticker, action, quantity)


    def execute_what_if_check(self, contract: Contract, order: Order) -> bool:
        """Runs a What-If order and validates the margin impact."""
        self.logger.info("Performing What-If margin check...")
        what_if_order = self._create_order(order.action, order.totalQuantity, order.orderType)
        what_if_order.whatIf = True
        self.what_if_result = None

        what_if_id = self.get_next_order_id()
        self.open_order_end_event.clear()
        self.placeOrder(what_if_id, contract, what_if_order)

        if not self.open_order_end_event.wait(10):
            self.logger.error("Timeout on What-If check response.")
            return False

        if self.what_if_result is None or not self.what_if_result.get('initMargin'):
            self.logger.error("Did not receive valid margin data from What-If check.")
            return False

        self.logger.info(f"What-If check successful. Margin impact: {self.what_if_result}")
        return True


    def monitor_and_close_loop(self, ticker: str, original_action: str, quantity: float):
        """Keeps the script alive and triggers the closing order at the right time."""
        today = pd.Timestamp.now(tz=config.TIMEZONE).normalize()
        try:
            schedule = get_market_schedule_for_date(today, get_lse_calendar())
            _, close_ts = _validate_schedule(schedule)
        except Exception as e:
            self.logger.error(f"Could not get market schedule to plan close: {e}")
            return

        submission_time = (close_ts - pd.Timedelta(minutes=config.MOC_SUBMISSION_WINDOW_MINUTES)).to_pydatetime()
        self.logger.info(f"Market closes at {close_ts.strftime('%H:%M:%S')}. MOC order will be sent at {submission_time.strftime('%H:%M:%S')}.")

        _safe_sleep_until(submission_time, config.TIMEZONE)

        if not SHUTDOWN_EVENT.is_set():
            self.close_position(ticker, original_action, quantity)

        self.logger.info("Closing order sent. Shutting down in 60 seconds.")
        time.sleep(60)


    def close_position(self, ticker: str, original_action: str, quantity: float):
        """Places a Market-on-Close order to close the day's position with a fallback."""
        if quantity <= 0:
            self.logger.error("close_position called with non-positive quantity; aborting.")
            return

        closing_action = "SELL" if original_action == "BUY" else "BUY"
        self.logger.info(f"Placing CLOSING MOC order for {quantity} {ticker}...")

        closing_contract = self._create_contract(ticker)
        moc_order = self._create_order(closing_action, quantity, order_type="MOC")
        moc_order_id = self.get_next_order_id()
        self.placeOrder(moc_order_id, closing_contract, moc_order)

        time.sleep(5)

        if self.open_orders.get(moc_order_id, {}).get('status') == 'Cancelled':
            self.logger.warning("MOC order was rejected/cancelled! Attempting fallback with MKT order.")
            mkt_order = self._create_order(closing_action, quantity, order_type="MKT")
            self.placeOrder(self.get_next_order_id(), closing_contract, mkt_order)

    def request_opening_prices(self):
        """Requests snapshot of opening prices for all TRADABLE tickers."""
        if not self.tradable_tickers:
            self.logger.warning("No tradable tickers to request opening prices for.")
            return
            
        self.logger.info(f"Requesting opening prices for {len(self.tradable_tickers)} tradable stocks...")
        
        # Rebuild the request ID maps based on the filtered list of tradable tickers
        self.open_prices = {}
        self.open_price_req_ids = {i + 1: ticker for i, ticker in enumerate(self.tradable_tickers)}
        self.open_price_ticker_map = {v: k for k, v in self.open_price_req_ids.items()}
        
        for ticker in self.tradable_tickers:
            reqId = self.open_price_ticker_map[ticker]
            contract = self._create_contract(ticker)
            try:
                # Request a single snapshot (snapshot=True)
                self.reqMktData(reqId, contract, "", True, False, [])
            except Exception as e:
                self.logger.error(f"reqMktData failed for {ticker}: {e}")

        start_time = time.time()
        timeout = config.OPENING_PRICE_TIMEOUT_SECONDS
        while (time.time() - start_time < timeout) and not SHUTDOWN_EVENT.is_set():
            if len(self.open_prices) == len(self.tradable_tickers):
                self.logger.info("All opening prices for tradable tickers received.")
                break
            time.sleep(1)
        else:
            self.logger.warning(
                f"Finished opening price collection after {timeout}s. "
                f"Received {len(self.open_prices)}/{len(self.tradable_tickers)} prices."
            )
        
        # Cancel any requests that didn't get a response
        for req_id in self.open_price_req_ids:
            if self.open_price_req_ids[req_id] not in self.open_prices:
                self.cancelMktData(req_id)


    def get_next_order_id(self) -> int:
        with self._id_lock:
            if self.next_order_id is None:
                raise RuntimeError("next_order_id is not initialized yet.")
            current_id = self.next_order_id
            self.next_order_id += 1
            return current_id

    def shutdown(self):
        self.logger.info("Shutting down bot.")
        SHUTDOWN_EVENT.set()
        time.sleep(1) # Give threads a moment to see the event
        if self.isConnected():
            self.disconnect()


def main():
    """Main entry point for the bot."""
    try:
        _validate_config_and_env()
    except Exception as e:
        print(f"[FATAL] Configuration/Environment error: {e}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logging(Path(config.BASE_DIR) / "logs")

    def _handle_signal(signum, frame):
        logger.info(f"Received signal {signum}. Shutting down gracefully.")
        SHUTDOWN_EVENT.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    parser = argparse.ArgumentParser(description="LSE Algorithmic Trading Bot")
    parser.add_argument("mode", choices=['trading', 'overnight'], help="The mode to run the bot in.")
    args = parser.parse_args()

    lse_calendar = get_lse_calendar()
    today = pd.Timestamp.now(tz=config.TIMEZONE).normalize()

    try:
        schedule = get_market_schedule_for_date(today, lse_calendar)
        is_trading_day = schedule is not None
    except Exception as e:
        logger.error(f"Failed to get market schedule: {e}")
        is_trading_day = False

    if not is_trading_day and args.mode == 'trading':
        logger.info("Today is not a trading day. Exiting.")
        return

    try:
        tickers = _load_tickers(Path(config.TICKER_FILE))
    except Exception as e:
        logger.critical(f"Failed to load tickers: {e}")
        return

    app = None
    try:
        app = IBApp(tickers)
        app.connect_and_start()

        if args.mode == 'trading':
            logger.info("--- Starting TRADING session ---")
            app.sync_state()
            app.prepare_tradable_tickers()

            if not app.tradable_tickers:
                logger.critical("No tradable tickers found. Shutting down.")
            else:
                open_ts, _ = _validate_schedule(schedule)
                _safe_sleep_until(open_ts.to_pydatetime(), config.TIMEZONE)
                
                if not SHUTDOWN_EVENT.is_set():
                    app.run_trading_session()

        elif args.mode == 'overnight':
            logger.info("--- Starting OVERNIGHT data update ---")
            app.run_overnight_session()

    except (ConnectionError, TimeoutError, ValueError) as e:
        logger.critical(f"A connection or setup error occurred: {e}")
    except KeyboardInterrupt:
        logger.info("Bot stopped manually by user.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
    finally:
        if app and app.isConnected():
            app.shutdown()


if __name__ == "__main__":
    main()
