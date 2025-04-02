"""
Technical indicator implementations for vaex dataframes.
This replaces pandas_ta functionality with vaex-compatible implementations.
"""

import vaex
import numpy as np

def ema(df, column='close', length=10):
    """Calculate Exponential Moving Average (EMA) for a vaex dataframe"""
    # Vaex doesn't have built-in EMA, so we implement it manually
    alpha = 2.0 / (length + 1)
    
    # Convert to numpy for calculation
    values = df[column].values
    ema_values = np.zeros_like(values)
    
    # Initialize with SMA
    ema_values[:length] = np.mean(values[:length])
    
    # Calculate EMA
    for i in range(length, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i-1]
    
    return ema_values

def sma(df, column='close', length=10):
    """Simple Moving Average using vaex's rolling method"""
    return df[column].rolling(window=length).mean()

def rsi(df, column='close', length=14):
    """Calculate Relative Strength Index (RSI) for vaex dataframe"""
    # Calculate price changes
    delta = df[column].diff()
    
    # Get gains and losses
    gains = np.where(delta.values > 0, delta.values, 0)
    losses = np.where(delta.values < 0, -delta.values, 0)
    
    # Initialize arrays for avg_gain and avg_loss
    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)
    
    # First average is simple average
    avg_gains[length] = np.mean(gains[1:length+1])
    avg_losses[length] = np.mean(losses[1:length+1])
    
    # Calculate smoothed averages
    for i in range(length+1, len(gains)):
        avg_gains[i] = (avg_gains[i-1] * (length-1) + gains[i]) / length
        avg_losses[i] = (avg_losses[i-1] * (length-1) + losses[i]) / length
    
    # Calculate RS and RSI
    rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses!=0)
    rsi_values = 100 - (100 / (1 + rs))
    
    # First length values are NaN
    rsi_values[:length] = np.nan
    
    return rsi_values

def macd(df, column='close', fast=12, slow=26, signal=9):
    """Calculate MACD indicator for vaex dataframe"""
    # Calculate EMAs
    fast_ema_values = ema(df, column, fast)
    slow_ema_values = ema(df, column, slow)
    
    # MACD line = fast EMA - slow EMA
    macd_line = fast_ema_values - slow_ema_values
    
    # Signal line = EMA of MACD line
    # We create a temporary dataframe to calculate EMA of MACD
    temp_df = vaex.from_arrays(macd=macd_line)
    signal_line = ema(temp_df, 'macd', signal)
    
    # Histogram = MACD line - signal line
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def bollinger_bands(df, column='close', length=20, std_dev=2):
    """Calculate Bollinger Bands for vaex dataframe"""
    # Calculate middle band (SMA)
    middle_band = sma(df, column, length)
    
    # Calculate standard deviation
    rolling_std = df[column].rolling(window=length).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # Calculate bandwidth and %B
    bandwidth = (upper_band - lower_band) / middle_band
    
    # %B = (price - lower) / (upper - lower)
    # Need to handle potential division by zero
    price = df[column].values
    percent_b = np.divide(
        price - lower_band, 
        upper_band - lower_band,
        out=np.full_like(price, 0.5),
        where=(upper_band - lower_band) != 0
    )
    
    return {
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band,
        'bandwidth': bandwidth,
        'percent_b': percent_b
    }

def stochastic(df, k_period=14, d_period=3, slowing=3):
    """Calculate Stochastic Oscillator for vaex dataframe"""
    # Get min and max values over k_period
    low_min = df.rolling(window=k_period, on='low').min()
    high_max = df.rolling(window=k_period, on='high').max()
    
    # Calculate %K
    # %K = 100 * (close - lowest low) / (highest high - lowest low)
    close_values = df['close'].values
    k_values = 100 * np.divide(
        close_values - low_min,
        high_max - low_min,
        out=np.full_like(close_values, 50),
        where=(high_max - low_min) != 0
    )
    
    # Apply slowing if needed (slowing is a simple moving average of %K)
    if slowing > 1:
        # Create a temporary dataframe for the slowing calculation
        temp_df = vaex.from_arrays(k=k_values)
        k_values = sma(temp_df, 'k', slowing)
    
    # Calculate %D (SMA of %K)
    temp_df = vaex.from_arrays(k=k_values)
    d_values = sma(temp_df, 'k', d_period)
    
    return {
        'k': k_values,
        'd': d_values
    }

def atr(df, length=14):
    """Calculate Average True Range for vaex dataframe"""
    # Calculate true ranges
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    
    # True Range is max of: current high - current low, 
    # |current high - previous close|, |current low - previous close|
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    
    # True Range is the max of the three
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR using EMA of TR
    temp_df = vaex.from_arrays(tr=tr)
    atr_values = ema(temp_df, 'tr', length)
    
    return atr_values

def obv(df):
    """Calculate On-Balance Volume for vaex dataframe"""
    # Initialize OBV array
    obv_values = np.zeros(len(df))
    
    # Get close and volume values
    close = df['close'].values
    volume = df['volume'].values
    
    # First value is just the first volume
    obv_values[0] = volume[0]
    
    # Calculate OBV based on price movement
    for i in range(1, len(df)):
        if close[i] > close[i-1]:
            # Price up, add volume
            obv_values[i] = obv_values[i-1] + volume[i]
        elif close[i] < close[i-1]:
            # Price down, subtract volume
            obv_values[i] = obv_values[i-1] - volume[i]
        else:
            # Price unchanged, OBV unchanged
            obv_values[i] = obv_values[i-1]
    
    return obv_values
