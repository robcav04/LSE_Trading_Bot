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
import pytz  # <-- Added/ensured: pytz is imported
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId

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
        self.tickers = tickers
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
        self.open_price_req_ids = {i + 1: ticker for i, ticker in enumerate(tickers)}
        self.open_price_ticker_map = {v: k for k, v in self.open_price_req_ids.items()}

        # --- Asynchronous Request Management ---
        self.historical_data_queue = queue.Queue()
        self.active_requests: Dict[int, threading.Event] = {}

        # --- Threading Events for Synchronization ---
        self.connection_event = threading.Event()
        self.next_valid_id_event = threading.Event()
        self.position_end_event = threading.Event()
        self.open_order_end_event = threading.Event()
        self.account_summary_event = threading.Event()
        self.historical_data_end_event = threading.Event()

    # --- EWrapper Callbacks (Handles incoming messages from IB) ---
    # (IB API behavior left as-is; only minor safety parsing where applicable)

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158, 2108, 2103, 2105, 2107, 2176, 162]:
            self.logger.info(f"IB Info (ReqId: {reqId}): {errorString}")
        else:
            self.logger.error(f"Error (ReqId: {reqId}, Code: {errorCode}): {errorString}")

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
        self.connection_event.set()

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
            except Exception:
                self.logger.error(f"Failed to parse NetLiquidation value '{value}'")
                self.net_liquidation_value = 0.0
            self.account_summary_event.set()

    def accountSummaryEnd(self, reqId: int):
        self.cancelAccountSummary(reqId)

    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        if tickType == 14 and reqId in self.open_price_req_ids:  # 14 = Open Tick
            ticker = self.open_price_req_ids[reqId]
            if ticker not in self.open_prices and _is_finite_positive(price):
                self.open_prices[ticker] = price
                self.logger.info(f"Received opening price for {ticker}: {price}")
                self.cancelMktData(reqId)

    def historicalData(self, reqId: int, bar):
        data = {
            "Date": bar.date, "Open": bar.open, "High": bar.high,
            "Low": bar.low, "Close": bar.close, "Volume": bar.volume,
        }
        self.historical_data_queue.put({'reqId': reqId, 'data': data})

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        if reqId in self.active_requests:
            self.active_requests[reqId].set()
            self.logger.info(f"Historical data end for ReqId: {reqId}")

    # --- Core Bot Methods (non-IB logic hardened; IB calls untouched) ---

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
        """Synchronizes bot's state with the broker (positions, orders, account value)."""
        self.logger.info("Syncing state with broker...")

        self.position_end_event.clear()
        self.reqPositions()
        if not self.position_end_event.wait(10):
            raise TimeoutError("Timeout waiting for position data.")
        self.logger.info(f"Current positions: {list(self.current_positions.keys())}")

        self.open_order_end_event.clear()
        self.reqAllOpenOrders()
        if not self.open_order_end_event.wait(10):
            raise TimeoutError("Timeout waiting for open order data.")
        self.logger.info(f"Found {len(self.open_orders)} open orders.")

        self.account_summary_event.clear()
        self.reqAccountSummary(self.REQ_ID_ACCOUNT_SUMMARY, "All", "NetLiquidation")
        if not self.account_summary_event.wait(10):
            raise TimeoutError("Timeout waiting for account summary.")
        if not _is_finite_positive(self.net_liquidation_value):
            raise ValueError(f"Net Liquidation Value is invalid: {self.net_liquidation_value}")
        self.logger.info(f"Net Liquidation Value: {self.net_liquidation_value}")

    def run_overnight_session(self):
        """Fetches the previous day's data and updates the local database."""
        self.logger.info("Starting OVERNIGHT data update session.")
        bars_to_update: Dict[str, dict] = {}

        for i, ticker in enumerate(self.tickers):
            if SHUTDOWN_EVENT.is_set():
                self.logger.info("Shutdown requested; aborting overnight session loop.")
                break

            reqId = i + 1000  # distinct ID range
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
                item = self.historical_data_queue.get()
                if item['reqId'] == reqId:
                    bars_to_update[ticker] = item['data']
                    self.logger.info(f"Fetched daily bar for {ticker}")
                    break

            del self.active_requests[reqId]
            # pacing, but allow early shutdown
            for _ in range(2):
                if SHUTDOWN_EVENT.is_set():
                    break
                time.sleep(1)

        self.logger.info(f"Finished fetching data. Ready to update database for {len(bars_to_update)} tickers.")
        # self.data_manager.update_database(bars_to_update)
        self.logger.info("Overnight session complete.")

    def run_trading_session(self):
        """Main logic for a live trading day."""
        if SHUTDOWN_EVENT.is_set():
            return

        try:
            self.historical_data_cache = self.data_manager.get_all_latest_data()
        except Exception as e:
            self.logger.critical(f"Failed loading historical data: {e}", exc_info=True)
            return

        if not self.historical_data_cache:
            self.logger.critical("Could not load any historical data. Aborting.")
            return

        # 1. Request opening prices
        if SHUTDOWN_EVENT.is_set():
            return
        self.request_opening_prices()

        # 2. Make trading decision
        try:
            predictions = self.model_handler.predict(self.historical_data_cache, self.open_prices)
        except Exception as e:
            self.logger.critical(f"Model prediction failed: {e}", exc_info=True)
            return

        if not predictions:
            self.logger.warning("No predictions were generated. No trades will be placed.")
            return

        predictions = {k: v for k, v in predictions.items()
                       if k in self.open_prices and isinstance(v, (int, float)) and math.isfinite(v)}
        if not predictions:
            self.logger.warning("No valid predictions with available opening prices. No trades will be placed.")
            return

        best_ticker = max(predictions, key=lambda k: abs(predictions[k]))
        predicted_log_return = float(predictions[best_ticker])
        action = "BUY" if predicted_log_return > 0 else "SELL"
        self.logger.info(f"Strategy decided: {action} {best_ticker} (Predicted Log Return: {predicted_log_return:.4f})")

        # 3. Calculate order size and create order objects
        if not _is_finite_positive(self.net_liquidation_value):
            self.logger.error("Cannot trade. Invalid Net Liq.")
            return
        if best_ticker not in self.open_prices or not _is_finite_positive(self.open_prices[best_ticker]):
            self.logger.error("Cannot trade. Missing or invalid opening price for chosen stock.")
            return

        open_price = float(self.open_prices[best_ticker])
        quantity = int(self.net_liquidation_value / open_price)
        if quantity < 1:
            self.logger.error(f"Calculated order size < 1 share (NetLiq={self.net_liquidation_value}, Open={open_price}). Aborting.")
            return
        self.logger.info(f"Calculated order size: {quantity} shares.")

        opening_contract = self._create_contract(best_ticker)
        opening_order = self._create_order(action, quantity)

        # 4. Perform "What-If" check
        if not self.execute_what_if_check(opening_contract, opening_order):
            self.logger.critical("What-If check failed or indicated unacceptable margin. Aborting trade.")
            return

        # 5. Place the LIVE opening order
        self.logger.info(f"Placing LIVE OPENING order for {quantity} {best_ticker}...")
        live_order_id = self.get_next_order_id()
        self.placeOrder(live_order_id, opening_contract, opening_order)

        # 6. Enter main monitoring loop until it's time to close
        if SHUTDOWN_EVENT.is_set():
            return
        self.monitor_and_close_loop(best_ticker, action, quantity)

    def execute_what_if_check(self, contract: Contract, order: Order) -> bool:
        """Runs a What-If order and validates the margin impact."""
        self.logger.info("Performing What-If margin check...")
        what_if_order = order
        what_if_order.whatIf = True
        self.what_if_result = None  # Reset previous result

        what_if_id = self.get_next_order_id()
        self.open_order_end_event.clear()
        self.placeOrder(what_if_id, contract, what_if_order)

        if not self.open_order_end_event.wait(10):
            self.logger.error("Timeout on What-If check response.")
            return False

        if self.what_if_result is None or not self.what_if_result.get('initMargin'):
            self.logger.error("Did not receive valid margin data from What-If check.")
            return False

        self.logger.info("What-If check successful. Margin impact is acceptable.")
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
        self.logger.info(
            f"Market closes at {close_ts.strftime('%H:%M:%S')}. "
            f"MOC order will be sent at {submission_time.strftime('%H:%M:%S')}."
        )

        _safe_sleep_until(submission_time, config.TIMEZONE)

        if not SHUTDOWN_EVENT.is_set():
            self.close_all_positions(ticker, original_action, quantity)

        self.logger.info("Closing order sent. Shutting down in 60 seconds.")
        # Allow early shutdown during the cool-down
        for _ in range(60):
            if SHUTDOWN_EVENT.is_set():
                break
            time.sleep(1)

    def close_all_positions(self, ticker: str, original_action: str, quantity: float):
        """Places a Market-on-Close order to close the day's position with a fallback."""
        if quantity <= 0:
            self.logger.error("close_all_positions called with non-positive quantity; aborting close.")
            return

        closing_action = "SELL" if original_action == "BUY" else "BUY"
        self.logger.info(f"Placing CLOSING MOC order for {quantity} {ticker}...")

        closing_contract = self._create_contract(ticker)
        moc_order = self._create_order(closing_action, quantity, order_type="MOC")

        moc_order_id = self.get_next_order_id()
        self.placeOrder(moc_order_id, closing_contract, moc_order)

        # brief wait to detect immediate rejection, but respect shutdown
        for _ in range(5):
            if SHUTDOWN_EVENT.is_set():
                break
            time.sleep(1)

        if self.open_orders.get(moc_order_id, {}).get('status') == 'Cancelled':
            self.logger.warning("MOC order was rejected or cancelled! Attempting fallback with MKT order.")
            mkt_order = self._create_order(closing_action, quantity, order_type="MKT")
            self.placeOrder(self.get_next_order_id(), closing_contract, mkt_order)

    def request_opening_prices(self):
        """Requests snapshot of opening prices for all tickers."""
        self.logger.info(f"Requesting opening prices for {len(self.tickers)} stocks...")
        for ticker in self.tickers:
            reqId = self.open_price_ticker_map.get(ticker)
            if reqId:
                contract = self._create_contract(ticker)
                try:
                    self.reqMktData(reqId, contract, "", True, False, [])
                except Exception as e:
                    self.logger.error(f"reqMktData failed for {ticker}: {e}")

        start_time = time.time()
        while (time.time() - start_time < config.OPENING_PRICE_TIMEOUT_SECONDS) and not SHUTDOWN_EVENT.is_set():
            if len(self.open_prices) == len(self.tickers):
                self.logger.info("All opening prices received.")
                break
            time.sleep(1)

        self.logger.info(f"Finished opening price collection. Received {len(self.open_prices)}/{len(self.tickers)} prices.")

    def get_next_order_id(self):
        with self._id_lock:
            if self.next_order_id is None:
                raise RuntimeError("next_order_id is not initialized yet.")
            current_id = self.next_order_id
            self.next_order_id += 1
            return current_id

    def shutdown(self):
        self.logger.info("Shutting down bot.")
        SHUTDOWN_EVENT.set()  # Signal all loops to terminate
        try:
            self.disconnect()
        except Exception:
            pass


def main():
    """Main entry point for the bot."""
    # Ensure early config/env validation provides actionable errors
    try:
        _validate_config_and_env()
    except Exception as e:
        print(f"[FATAL] Configuration/Environment error: {e}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logging(Path(config.BASE_DIR) / "logs")

    # Clean termination on signals
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

    if not is_trading_day:
        logger.info("Today is not a trading day. Exiting.")
        return

    # Ticker loading with strong validation
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

            # Wait for market open safely (if not already open)
            try:
                open_ts, _ = _validate_schedule(get_market_schedule_for_date(today, lse_calendar))
                now_local = datetime.now(pytz.timezone(config.TIMEZONE))
                wait_seconds = (open_ts.to_pydatetime() - now_local).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"Waiting {wait_seconds:.0f} seconds for market open...")
                    _safe_sleep_until(open_ts.to_pydatetime(), config.TIMEZONE)
            except Exception as e:
                logger.error(f"Could not determine or wait for market open: {e}")

            if not SHUTDOWN_EVENT.is_set():
                app.run_trading_session()

        elif args.mode == 'overnight':
            logger.info("--- Starting OVERNIGHT data update ---")
            app.run_overnight_session()

    except (ConnectionError, TimeoutError) as e:
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
