import pandas as pd
import numpy as np


def _wilder_ema(series: pd.Series, length: int) -> pd.Series:
    """Calculates the Wilder's Exponential Moving Average."""
    # For Wilder's smoothing, alpha = 1 / length, so com = length - 1
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
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
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

    return pd.DataFrame({'STOCHk': stoch_k, 'STOCHd': stoch_d})


def roc(close: pd.Series, length: int = 12, **kwargs) -> pd.Series:
    """Rate of Change (ROC)"""
    return 100 * (close.diff(length) / close.shift(length))


def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, **kwargs) -> pd.Series:
    """Williams %R"""
    lowest_low = low.rolling(length).min()
    highest_high = high.rolling(length).max()

    willr_series = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return willr_series