from datetime import datetime, timedelta
import polars as pl
import numpy as np
import requests
import json
import time
import os
import traceback
from api_utils import CryptoAPIUtils
import pyarrow as pa  # Import pyarrow to fix the Parabolic SAR calculation error

class CryptoIndicators:
    def __init__(self, api_key="your_default_api_key_here", discord_logger=None):
        self.api_key = api_key
        self.data = None  # Will hold Polars DataFrame
        self.signals = {}
        self.discord_logger = discord_logger
        self.log_messages = []
        
        # Set up more comprehensive indicator weights
        self.indicator_weights = {
            'rsi': 1.0,
            'macd': 1.0,
            'ema': 1.0,
            'atr': 0.8,
            'volume': 0.9,
            'bollinger': 1.0,
            'stochastic': 0.9,
            'ichimoku': 0.8,
            'obv': 0.7,
            'parabolic_sar': 0.6
        }
        
        # Performance tracking for each indicator
        self.indicator_performance = {
            indicator: {
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.5,
                'last_updated': datetime.now().isoformat()
            } for indicator in self.indicator_weights
        }
        
        # Load previous performance if available
        self._load_performance()
        
        # Log initialization
        self.log("Initializing Crypto Indicators system with enhanced indicators")
        
        # Add API utils for free API access
        self.api_utils = CryptoAPIUtils()
        
    def log(self, message, level="INFO", send_to_discord=False):
        """Log a message and optionally send to Discord"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        self.log_messages.append(log_message)
        
        # Send to Discord if requested and discord_logger is available
        if send_to_discord and self.discord_logger:
            # Use send_log_sync to avoid async issues
            self.discord_logger.send_log_sync(message, level)
            
    def fetch_data(self, symbol='BTC/USD', timeframe='1h', limit=200):
        """Fetch cryptocurrency data from a real API"""
        self.log(f"Fetching {limit} {timeframe} candles for {symbol}...")
        
        try:
            # Use our free API utility instead of requiring API keys
            price_data = self.api_utils.get_crypto_price_history(
                symbol=symbol,
                days=int(limit / 24) + 1,
                interval='hourly' if timeframe in ['1h', '2h', '4h'] else 'daily'
            )
            
            if not price_data or not price_data.get('prices'):
                self.log("No data returned from free APIs, attempting fallback...", level="WARNING")
                return self._fetch_from_fallback(symbol, timeframe, limit)
            
            # Process the price data
            prices = price_data.get('prices', [])
            volumes = price_data.get('volumes', [])
            
            # Create Polars DataFrame with timestamp and close price
            df = pl.DataFrame({
                'timestamp': [p[0] for p in prices],
                'close': [p[1] for p in prices]
            })
            
            # Convert timestamp from milliseconds to datetime
            df = df.with_columns([
                pl.col('timestamp').cast(pl.Datetime).dt.with_time_unit('ms')
            ])
            
            # Add volume data if available
            if volumes:
                volume_df = pl.DataFrame({
                    'timestamp': [v[0] for v in volumes],
                    'volume': [v[1] for v in volumes]
                })
                volume_df = volume_df.with_columns([
                    pl.col('timestamp').cast(pl.Datetime).dt.with_time_unit('ms')
                ])
                df = df.join(volume_df, on='timestamp', how='left')
            else:
                df = df.with_columns([
                    pl.lit(0).alias('volume')
                ])
                
            # For OHLC data, approximate from close prices if needed
            if 'open' not in df.columns:
                # Create multiple columns at once using with_columns
                df = df.with_columns([
                    pl.col('close').shift(1).alias('open'),
                    (pl.col('close') * 1.005).alias('high'),
                    (pl.col('close') * 0.995).alias('low')
                ])
                
                # Fill NaN values in 'open' with 'close'
                df = df.with_columns([
                    pl.when(pl.col('open').is_null())
                    .then(pl.col('close'))
                    .otherwise(pl.col('open'))
                    .alias('open')
                ])
            
            # Keep only the most recent 'limit' records
            df = df.sort('timestamp', descending=True).head(limit).sort('timestamp')
            
            # Set the data
            self.data = df
            
            self.log(f"Successfully fetched {df.height} records from free API for {symbol}", send_to_discord=True)
            return True
            
        except Exception as e:
            self.log(f"Error fetching data: {e}", level="ERROR", send_to_discord=True)
            self.log("Attempting to use fallback data source...", level="WARNING")
            
            try:
                # Try an alternative data source as fallback
                return self._fetch_from_fallback(symbol, timeframe, limit)
            except Exception as e2:
                self.log(f"Fallback source also failed: {e2}", level="ERROR", send_to_discord=True)
                return False
    
    def run_analysis(self, symbol='BTC/USD', timeframe='1h', limit=100):
        """Run the complete indicator analysis pipeline with performance optimization"""
        self.log(f"Starting full technical analysis for {symbol} ({timeframe})", send_to_discord=True)
        start_time = time.time()
        
        if not self.fetch_data(symbol, timeframe, limit):
            self.log("Failed to fetch data, aborting analysis", level="ERROR", send_to_discord=True)
            return None
        
        # Calculate primary indicators
        try:
            # Core indicators first
            self.calculate_rsi()
            self.calculate_macd()
            self.calculate_ema([20, 50, 200])
            self.calculate_bollinger_bands()
            
            # Secondary indicators
            self.calculate_obv()
            
            # Additional indicators based on available memory and processing power
            self.analyze_volume()
            self.calculate_stochastic()
            self.calculate_atr()
            self.calculate_parabolic_sar()
            
            # This one is computationally expensive, can be disabled for performance
            self.calculate_ichimoku()
            
            # Generate signals
            signals = self.generate_signals()
            
            execution_time = time.time() - start_time
            self.log(f"Analysis completed in {execution_time:.2f} seconds", send_to_discord=True)
            
            return signals
            
        except Exception as e:
            self.log(f"Error in technical analysis: {e}", level="ERROR", send_to_discord=True)
            self.log(f"Error details: {traceback.format_exc()}", level="ERROR")
            return None
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        if self.data is None:
            return False
        
        try:
            # Calculate price changes
            delta = self.data.with_columns([
                pl.col('close').diff().alias('delta')
            ])
            
            # Split gains and losses
            gain = delta.with_columns([
                pl.when(pl.col('delta') > 0).then(pl.col('delta')).otherwise(0).alias('gain')
            ])
            loss = delta.with_columns([
                pl.when(pl.col('delta') < 0).then(-pl.col('delta')).otherwise(0).alias('loss')
            ])
            
            # Calculate average gain and loss using rolling window
            avg_gain = gain.with_columns([
                pl.col('gain').rolling_mean(window_size=period).alias('avg_gain')
            ])
            avg_loss = loss.with_columns([
                pl.col('loss').rolling_mean(window_size=period).alias('avg_loss')
            ])
            
            # Combine the calculated columns
            rsi_df = avg_gain.select(['timestamp', 'avg_gain']).join(
                avg_loss.select(['timestamp', 'avg_loss']), on='timestamp'
            )
            
            # Calculate relative strength and RSI
            rs = rsi_df.with_columns([
                (pl.col('avg_gain') / pl.when(pl.col('avg_loss') == 0).then(0.0001).otherwise(pl.col('avg_loss'))).alias('rs')
            ])
            rsi = rs.with_columns([
                (100 - (100 / (1 + pl.col('rs')))).alias('rsi')
            ])
            
            # Add RSI to original dataframe
            self.data = self.data.join(rsi.select(['timestamp', 'rsi']), on='timestamp', how='left')
            
            # Calculate RSI divergence
            self.data = self.data.with_columns([
                (pl.col('rsi').diff(3) / 3).alias('rsi_slope'),
                ((pl.col('close').diff(3) / 3) / pl.col('close').shift(3)).alias('price_slope')
            ])
            
            # Fill NaN values
            self.data = self.data.with_columns([
                pl.col('rsi_slope').fill_null(0),
                pl.col('price_slope').fill_null(0)
            ])
            
            # Bullish and bearish divergence
            self.data = self.data.with_columns([
                ((pl.col('rsi_slope') > 0) & (pl.col('price_slope') < 0)).alias('bullish_divergence'),
                ((pl.col('rsi_slope') < 0) & (pl.col('price_slope') > 0)).alias('bearish_divergence')
            ])
            
            self.log("RSI calculated with divergence detection")
            return True
        except Exception as e:
            self.log(f"Error calculating RSI: {e}", level="ERROR")
            return False
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        if self.data is None:
            return False
        
        try:
            # Calculate EMAs using exponential weighted mean
            ema_fast = self.data.with_columns([
                pl.col('close').ewm_mean(span=fast).alias('ema_fast')
            ])
            ema_slow = self.data.with_columns([
                pl.col('close').ewm_mean(span=slow).alias('ema_slow')
            ])
            
            # Combine the calculated columns
            macd_df = ema_fast.select(['timestamp', 'ema_fast']).join(
                ema_slow.select(['timestamp', 'ema_slow']), on='timestamp'
            )
            
            # Calculate MACD Line
            macd = macd_df.with_columns([
                (pl.col('ema_fast') - pl.col('ema_slow')).alias('macd')
            ])
            
            # Calculate Signal Line using EMA of MACD
            signal_line = macd.with_columns([
                pl.col('macd').ewm_mean(span=signal).alias('macd_signal')
            ])
            
            # Calculate MACD Histogram
            histogram = signal_line.with_columns([
                (pl.col('macd') - pl.col('macd_signal')).alias('macd_hist')
            ])
            
            # Add MACD columns to original dataframe
            self.data = self.data.join(
                histogram.select(['timestamp', 'macd', 'macd_signal', 'macd_hist']), 
                on='timestamp', how='left'
            )
            
            # Calculate MACD histogram slope for momentum
            self.data = self.data.with_columns([
                pl.col('macd_hist').diff(3).alias('macd_hist_slope')
            ])
            
            # Detect MACD crosses
            self.data = self.data.with_columns([
                ((pl.col('macd') > pl.col('macd_signal')) & 
                 (pl.col('macd').shift(1) <= pl.col('macd_signal').shift(1))).alias('macd_cross_up'),
                ((pl.col('macd') < pl.col('macd_signal')) & 
                 (pl.col('macd').shift(1) >= pl.col('macd_signal').shift(1))).alias('macd_cross_down')
            ])
            
            self.log("MACD calculated with crossover detection")
            return True
        except Exception as e:
            self.log(f"Error calculating MACD: {e}", level="ERROR")
            return False
            
    def calculate_ema(self, periods=[20, 50, 200]):
        """Calculate Exponential Moving Average for specified periods"""
        if self.data is None:
            return False
        
        try:
            # Calculate EMA for each specified period
            for period in periods:
                self.data = self.data.with_columns([
                    pl.col('close').ewm_mean(span=period).alias(f'ema_{period}')
                ])
                
            # Calculate EMA crosses (short term vs long term)
            if all(p in periods for p in [20, 50]):
                self.data = self.data.with_columns([
                    (pl.col('ema_20') > pl.col('ema_50')).alias('ema_20_above_50')
                ])
                
            if all(p in periods for p in [50, 200]):
                self.data = self.data.with_columns([
                    (pl.col('ema_50') > pl.col('ema_200')).alias('ema_50_above_200')
                ])
                
            self.log(f"EMA calculated for periods: {periods}")
            return True
            
        except Exception as e:
            self.log(f"Error calculating EMA: {e}", level="ERROR")
            return False

    def calculate_bollinger_bands(self, period=20, std_dev=2.0):
        """Calculate Bollinger Bands"""
        if self.data is None:
            return False
        
        try:
            # Calculate middle band (SMA)
            self.data = self.data.with_columns([
                pl.col('close').rolling_mean(window_size=period).alias('bb_middle')
            ])
            
            # Calculate standard deviation
            self.data = self.data.with_columns([
                pl.col('close').rolling_std(window_size=period).alias('bb_std')
            ])
            
            # Calculate upper and lower bands
            self.data = self.data.with_columns([
                (pl.col('bb_middle') + (pl.col('bb_std') * std_dev)).alias('bb_upper'),
                (pl.col('bb_middle') - (pl.col('bb_std') * std_dev)).alias('bb_lower')
            ])
            
            # Calculate %B and bandwidth
            self.data = self.data.with_columns([
                ((pl.col('close') - pl.col('bb_lower')) / 
                 (pl.col('bb_upper') - pl.col('bb_lower'))).alias('bb_percent_b'),
                
                ((pl.col('bb_upper') - pl.col('bb_lower')) / 
                 pl.col('bb_middle')).alias('bb_bandwidth')
            ])
            
            self.log("Bollinger Bands calculated")
            return True
            
        except Exception as e:
            self.log(f"Error calculating Bollinger Bands: {e}", level="ERROR")
            return False

    def calculate_obv(self):
        """Calculate On-Balance Volume (OBV)"""
        if self.data is None or 'volume' not in self.data.columns:
            return False
        
        try:
            # Create a sign series based on price movement
            self.data = self.data.with_columns([
                pl.when(pl.col('close') > pl.col('close').shift(1))
                .then(1)
                .when(pl.col('close') < pl.col('close').shift(1))
                .then(-1)
                .otherwise(0)
                .alias('price_direction')
            ])
            
            # Apply direction to volume
            self.data = self.data.with_columns([
                (pl.col('price_direction') * pl.col('volume')).alias('directed_volume')
            ])
            
            # Calculate cumulative sum for OBV
            self.data = self.data.with_columns([
                pl.col('directed_volume').cum_sum().alias('obv')
            ])
            
            # Calculate OBV EMA for signals
            self.data = self.data.with_columns([
                pl.col('obv').ewm_mean(span=20).alias('obv_ema')
            ])
            
            # Calculate OBV slope for divergence detection
            self.data = self.data.with_columns([
                pl.col('obv').diff(3).alias('obv_slope')
            ])
            
            self.log("OBV calculated")
            return True
            
        except Exception as e:
            self.log(f"Error calculating OBV: {e}", level="ERROR")
            return False
            
    def _load_performance(self):
        """Load indicator performance from a file if it exists."""
        performance_file = 'data/indicator_performance.json'
        if os.path.exists(performance_file):
            try:
                with open(performance_file, 'r') as f:
                    self.indicator_performance = json.load(f)
                self.log("Loaded indicator performance from file")
            except Exception as e:
                self.log(f"Error loading performance file: {e}", level="ERROR")
        else:
            self.log("No performance file found, using default values")
            
    def _fetch_from_fallback(self, symbol, timeframe, limit):
        """Last resort: use Yahoo Finance for daily data or generate synthetic data"""
        self.log("Using fallback data source", level="WARNING", send_to_discord=True)
        
        try:
            # For Bitcoin and Ethereum, we can try Yahoo Finance
            if 'BTC' in symbol.upper() or 'BITCOIN' in symbol.upper():
                yahoo_symbol = 'BTC-USD'
            elif 'ETH' in symbol.upper() or 'ETHEREUM' in symbol.upper():
                yahoo_symbol = 'ETH-USD'
            else:
                self.log(f"No fallback available for {symbol}, using synthetic data", level="WARNING")
                return self._generate_synthetic_data(symbol, timeframe, limit)
                
            url = f"https://query1.finance.yahoo.com/v7/finance/chart/{yahoo_symbol}"
            params = {
                'range': '3mo',
                'interval': '1d',
                'indicators': 'quote',
                'includeTimestamps': 'true'
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                self.log(f"Yahoo Finance API error: {response.status_code}", level="ERROR")
                return self._generate_synthetic_data(symbol, timeframe, limit)
                
            data = response.json()
            
            # Extract price data
            result = data.get('chart', {}).get('result', [])
            if not result:
                self.log("No data returned from Yahoo Finance API", level="ERROR")
                return self._generate_synthetic_data(symbol, timeframe, limit)
                
            quotes = result[0]
            timestamps = quotes.get('timestamp', [])
            quote_data = quotes.get('indicators', {}).get('quote', [{}])[0]
            
            # Create DataFrame using Polars
            df = pl.DataFrame({
                'timestamp': timestamps,
                'open': quote_data.get('open', []),
                'high': quote_data.get('high', []),
                'low': quote_data.get('low', []),
                'close': quote_data.get('close', []),
                'volume': quote_data.get('volume', [])
            })
            
            # Convert timestamp and handle nulls
            df = df.with_columns([
                pl.col('timestamp').cast(pl.Datetime).dt.with_time_unit('s')
            ])
            
            # Drop null rows
            df = df.drop_nulls()
            
            # Limit to requested number of rows
            df = df.tail(limit)
            
            # Set the data
            self.data = df
            
            self.log(f"Successfully fetched {df.height} records from Yahoo Finance for {yahoo_symbol}", send_to_discord=True)
            return True
            
        except Exception as e:
            self.log(f"Fallback to Yahoo Finance failed: {e}", level="ERROR")
            
            # As absolute last resort, generate synthetic data with warning
            self.log("GENERATING SYNTHETIC DATA AS LAST RESORT - NOT SUITABLE FOR REAL TRADING", 
                     level="ERROR", send_to_discord=True)
            
            # Generate synthetic data based on real Bitcoin price patterns
            return self._generate_synthetic_data(symbol, timeframe, limit)
    
    def _generate_synthetic_data(self, symbol, timeframe, limit):
        """Generate synthetic price data for testing when real data is unavailable"""
        self.log("Generating synthetic data as last resort", level="WARNING")
        
        # Create timestamps
        end_time = datetime.now()
        timestamps = [(end_time - timedelta(hours=i)) for i in range(limit)]
        timestamps.reverse()
        
        # Generate synthetic price starting at a realistic value
        base_price = 30000 if 'BTC' in symbol.upper() else 2000  # BTC or ETH typical prices
        
        # Create price with random walk and some volatility
        price = base_price
        prices = []
        for _ in range(limit):
            # Random walk with slight upward bias
            change_pct = np.random.normal(0.0001, 0.015)  # mean, std_dev
            price = price * (1 + change_pct)
            prices.append(price)
        
        # Generate synthetic volume
        volumes = [np.random.normal(1000, 300) for _ in range(limit)]
        
        # Create Polars DataFrame
        df = pl.DataFrame({
            'timestamp': timestamps,
            'close': prices,
            'volume': volumes
        })
        
        # Add OHLC data
        df = df.with_columns([
            pl.col('close').shift(1).alias('open'),
            (pl.col('close') * (1 + np.random.normal(0.005, 0.003))).alias('high'),
            (pl.col('close') * (1 - np.random.normal(0.005, 0.003))).alias('low')
        ])
        
        # Fill nulls in the first row
        df = df.with_columns([
            pl.when(pl.col('open').is_null())
            .then(pl.col('close'))
            .otherwise(pl.col('open'))
            .alias('open')
        ])
        
        # Set the data
        self.data = df
        
        self.log(f"Created synthetic data with {df.height} records for {symbol}", level="WARNING", send_to_discord=True)
        return True
        
    def analyze_volume(self):
        """Analyze volume patterns and trends"""
        if self.data is None:
            return False
            
        try:
            # Calculate volume moving average
            self.data = self.data.with_columns([
                pl.col('volume').rolling_mean(window_size=20).alias('volume_sma_20')
            ])
            
            # Calculate volume change percentage
            self.data = self.data.with_columns([
                (pl.col('volume') / pl.col('volume_sma_20') - 1).alias('volume_pct_change')
            ])
            
            # Volume spikes (when volume > 2x its moving average)
            self.data = self.data.with_columns([
                (pl.col('volume') > pl.col('volume_sma_20') * 2).alias('volume_spike')
            ])
            
            self.log("Volume analysis completed")
            return True
        except Exception as e:
            self.log(f"Error analyzing volume: {e}", level="ERROR")
            return False

    def calculate_stochastic(self, k_period=14, d_period=3, slowing=3):
        """Calculate Stochastic Oscillator"""
        if self.data is None:
            return False
            
        try:
            rolling_min = self.data.with_columns([
                pl.col('low').rolling_min(window_size=k_period).alias('rolling_min')
            ])
            rolling_max = self.data.with_columns([
                pl.col('high').rolling_max(window_size=k_period).alias('rolling_max')
            ])
            
            # Merge back to main dataframe
            self.data = self.data.join(
                rolling_min.select(['timestamp', 'rolling_min']), 
                on='timestamp', how='left'
            )
            self.data = self.data.join(
                rolling_max.select(['timestamp', 'rolling_max']), 
                on='timestamp', how='left'
            )
            
            # Calculate %K
            self.data = self.data.with_columns([
                (100 * (pl.col('close') - pl.col('rolling_min')) / 
                 (pl.col('rolling_max') - pl.col('rolling_min') + 0.0001)).alias('stoch_k_raw')
            ])
            
            # Apply slowing (SMA of %K)
            self.data = self.data.with_columns([
                pl.col('stoch_k_raw').rolling_mean(window_size=slowing).alias('stoch_k')
            ])
            
            # Calculate %D (SMA of %K)
            self.data = self.data.with_columns([
                pl.col('stoch_k').rolling_mean(window_size=d_period).alias('stoch_d')
            ])
            
            self.log("Stochastic oscillator calculated")
            return True
        except Exception as e:
            self.log(f"Error calculating stochastic oscillator: {e}", level="ERROR")
            return False

    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        if self.data is None:
            return False
            
        try:
            # Calculate true range components
            self.data = self.data.with_columns([
                (pl.col('high') - pl.col('low')).alias('tr1'),
                (pl.col('high') - pl.col('close').shift(1)).abs().alias('tr2'),
                (pl.col('low') - pl.col('close').shift(1)).abs().alias('tr3')
            ])
            
            # Calculate true range as max of components
            self.data = self.data.with_columns([
                pl.max_horizontal([
                    pl.col('tr1'), 
                    pl.col('tr2'), 
                    pl.col('tr3')
                ]).alias('tr')
            ])
            
            # Calculate ATR as average of true range
            self.data = self.data.with_columns([
                pl.col('tr').rolling_mean(window_size=period).alias('atr')
            ])
            
            # Calculate ATR as percentage of price
            self.data = self.data.with_columns([
                (pl.col('atr') / pl.col('close') * 100).alias('atr_percent')
            ])
            
            self.log("ATR calculated")
            return True
        except Exception as e:
            self.log(f"Error calculating ATR: {e}", level="ERROR")
            return False

    def calculate_parabolic_sar(self, af_start=0.02, af_step=0.02, af_max=0.2):
        """Calculate Parabolic SAR"""
        if self.data is None or len(self.data) < 10:
            return False
            
        try:
            # Convert to pandas for this complex calculation (will return to polars after)
            import pandas as pd
            df = self.data.to_pandas()
            
            # Initialize variables
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            sar = np.zeros_like(close)
            trend = np.zeros_like(close)
            ep = np.zeros_like(close)
            af = np.zeros_like(close)
            
            # Initialize first SAR value
            trend[0] = 1  # Start with uptrend
            ep[0] = high[0]
            sar[0] = low[0]
            af[0] = af_start
            
            # Calculate SAR for each point
            for i in range(1, len(df)):
                # Update extreme point if needed
                if trend[i-1] > 0:  # Uptrend
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_step, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                else:  # Downtrend
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_step, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                
                # Calculate SAR
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # Ensure SAR is not within the current candle
                if trend[i-1] > 0:  # In uptrend, SAR must be below lows
                    sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                else:  # In downtrend, SAR must be above highs
                    sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                
                # Determine new trend
                if sar[i] < high[i]:
                    trend[i] = 1  # Uptrend
                else:
                    trend[i] = -1  # Downtrend
                    
                # If trend changed, reset AF and flip the SAR
                if trend[i] != trend[i-1]:
                    af[i] = af_start
                    if trend[i] > 0:
                        sar[i] = min(low[i], low[i-1])  # New uptrend, put SAR below lows
                        ep[i] = high[i]  # New EP is current high
                    else:
                        sar[i] = max(high[i], high[i-1])  # New downtrend, put SAR above highs
                        ep[i] = low[i]  # New EP is current low
            
            # Add results back to dataframe
            df['parabolic_sar'] = sar
            df['parabolic_sar_trend'] = trend
            
            # Back to polars
            self.data = pl.from_pandas(df)
            
            self.log("Parabolic SAR calculated")
            return True
        except Exception as e:
            self.log(f"Error calculating Parabolic SAR: {e}", level="ERROR")
            return False

    def calculate_ichimoku(self):
        """Calculate Ichimoku Cloud"""
        if self.data is None:
            return False
            
        try:
            # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
            tenkan_high = self.data.with_columns([
                pl.col('high').rolling_max(window_size=9).alias('tenkan_high')
            ])
            tenkan_low = self.data.with_columns([
                pl.col('low').rolling_min(window_size=9).alias('tenkan_low')
            ])
            
            # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
            kijun_high = self.data.with_columns([
                pl.col('high').rolling_max(window_size=26).alias('kijun_high')
            ])
            kijun_low = self.data.with_columns([
                pl.col('low').rolling_min(window_size=26).alias('kijun_low')
            ])
            
            # Merge components
            self.data = self.data.join(
                tenkan_high.select(['timestamp', 'tenkan_high']), 
                on='timestamp', how='left'
            )
            self.data = self.data.join(
                tenkan_low.select(['timestamp', 'tenkan_low']), 
                on='timestamp', how='left'
            )
            self.data = self.data.join(
                kijun_high.select(['timestamp', 'kijun_high']), 
                on='timestamp', how='left'
            )
            self.data = self.data.join(
                kijun_low.select(['timestamp', 'kijun_low']), 
                on='timestamp', how='left'
            )
            
            # Calculate lines
            self.data = self.data.with_columns([
                ((pl.col('tenkan_high') + pl.col('tenkan_low')) / 2).alias('ichimoku_tenkan'),
                ((pl.col('kijun_high') + pl.col('kijun_low')) / 2).alias('ichimoku_kijun')
            ])
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
            self.data = self.data.with_columns([
                ((pl.col('ichimoku_tenkan') + pl.col('ichimoku_kijun')) / 2).alias('ichimoku_senkou_a')
            ])
            
            # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for past 52 periods, shifted 26 periods ahead
            senkou_b_high = self.data.with_columns([
                pl.col('high').rolling_max(window_size=52).alias('senkou_b_high')
            ])
            senkou_b_low = self.data.with_columns([
                pl.col('low').rolling_min(window_size=52).alias('senkou_b_low')
            ])
            
            self.data = self.data.join(
                senkou_b_high.select(['timestamp', 'senkou_b_high']), 
                on='timestamp', how='left'
            )
            self.data = self.data.join(
                senkou_b_low.select(['timestamp', 'senkou_b_low']), 
                on='timestamp', how='left'
            )
            
            self.data = self.data.with_columns([
                ((pl.col('senkou_b_high') + pl.col('senkou_b_low')) / 2).alias('ichimoku_senkou_b')
            ])
            
            # Signal generation based on Tenkan/Kijun cross
            self.data = self.data.with_columns([
                (pl.col('ichimoku_tenkan') > pl.col('ichimoku_kijun')).alias('ichimoku_bullish'),
                (pl.col('ichimoku_tenkan') < pl.col('ichimoku_kijun')).alias('ichimoku_bearish'),
                (pl.col('close') > pl.max_horizontal([pl.col('ichimoku_senkou_a'), pl.col('ichimoku_senkou_b')])).alias('above_cloud'),
                (pl.col('close') < pl.min_horizontal([pl.col('ichimoku_senkou_a'), pl.col('ichimoku_senkou_b')])).alias('below_cloud')
            ])
            
            self.log("Ichimoku Cloud calculated")
            return True
        except Exception as e:
            self.log(f"Error calculating Ichimoku Cloud: {e}", level="ERROR")
            return False

    def generate_signals(self):
        """Generate trading signals based on calculated indicators"""
        if self.data is None:
            self.log("No data available for signal generation", level="ERROR")
            return None
        
        try:
            # Extract the latest values for indicators
            latest = self.data.tail(1)
            
            signals = {
                'timestamp': datetime.now().isoformat(),
                'price': float(latest['close'].iloc[0]),
                'indicators': {},
                'signal': 'neutral',
                'confidence': 0.5
            }
            
            # Extract indicator values
            indicators = {}
            
            # Get columns from the dataframe
            columns = self.data.columns
            
            # RSI
            if 'rsi' in columns:
                rsi_value = float(latest['rsi'])
                indicators['rsi'] = rsi_value
                indicators['rsi_signal'] = 1 if rsi_value < 30 else (-1 if rsi_value > 70 else 0)
            
            # MACD
            if all(col in columns for col in ['macd', 'macd_signal']):
                indicators['macd'] = float(latest['macd'])
                indicators['macd_signal_value'] = float(latest['macd_signal'])
                indicators['macd_hist'] = float(latest['macd']) - float(latest['macd_signal'])
                indicators['macd_signal'] = 1 if indicators['macd_hist'] > 0 else (-1 if indicators['macd_hist'] < 0 else 0)
            
            # EMA
            for period in [20, 50, 200]:
                ema_col = f'ema_{period}'
                if ema_col in columns:
                    indicators[ema_col] = float(latest[ema_col])
            
            # EMA crosses
            if all(col in columns for col in ['ema_20', 'ema_50']):
                indicators['ema_signal'] = 1 if float(latest['ema_20']) > float(latest['ema_50']) else -1
            
            # Bollinger Bands
            if all(col in columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                indicators['bb_upper'] = float(latest['bb_upper'])
                indicators['bb_lower'] = float(latest['bb_lower'])
                indicators['bb_middle'] = float(latest['bb_middle'])
                
                close_price = float(latest['close'])
                indicators['bollinger_signal'] = 1 if close_price < float(latest['bb_lower']) else (-1 if close_price > float(latest['bb_upper']) else 0)
            
            # Volume
            if 'volume' in columns:
                indicators['volume'] = float(latest['volume'])
                
                if 'volume_pct_change' in columns:
                    indicators['volume_change_pct'] = float(latest['volume_pct_change']) 
                    indicators['volume_signal'] = 1 if indicators['volume_change_pct'] > 0.5 else (-1 if indicators['volume_change_pct'] < -0.5 else 0)
            
            # OBV
            if 'obv' in columns:
                indicators['obv'] = float(latest['obv'])
                indicators['obv_signal'] = 0  # Default
                
                if 'obv_slope' in columns:
                    indicators['obv_slope'] = float(latest['obv_slope'])
                    indicators['obv_signal'] = 1 if float(latest['obv_slope']) > 0 else (-1 if float(latest['obv_slope']) < 0 else 0)
            
            # Stochastic
            if all(col in columns for col in ['stoch_k', 'stoch_d']):
                indicators['stoch_k'] = float(latest['stoch_k'])
                indicators['stoch_d'] = float(latest['stoch_d'])
                stoch_k = float(latest['stoch_k'])
                indicators['stoch_signal'] = 1 if stoch_k < 20 else (-1 if stoch_k > 80 else 0)
            
            # ATR
            if 'atr' in columns:
                indicators['atr'] = float(latest['atr'])
                if 'atr_percent' in columns:
                    indicators['atr_percent'] = float(latest['atr_percent'])
                
            # Parabolic SAR
            if 'parabolic_sar_trend' in columns:
                indicators['parabolic_sar_signal'] = int(latest['parabolic_sar_trend'])
            
            # Ichimoku
            if all(col in columns for col in ['ichimoku_tenkan', 'ichimoku_kijun']):
                indicators['ichimoku_tenkan'] = float(latest['ichimoku_tenkan'])
                indicators['ichimoku_kijun'] = float(latest['ichimoku_kijun'])
                indicators['ichimoku_signal'] = 1 if float(latest['ichimoku_tenkan']) > float(latest['ichimoku_kijun']) else -1
            
            signals['indicators'] = indicators
            
            # Calculate weighted signal based on all indicators
            signal_values = []
            weights = []
            
            # Add signals with their respective weights
            for indicator, weight in self.indicator_weights.items():
                signal_key = f"{indicator}_signal" 
                if signal_key in indicators:
                    signal_values.append(indicators[signal_key] * weight)
                    weights.append(weight)
            
            # Calculate final signal
            if weights:
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_signal = sum(signal_values) / total_weight
                    
                    # Determine signal direction and confidence
                    if weighted_signal > 0.2:
                        signals['signal'] = 'buy'
                        signals['confidence'] = min(0.5 + weighted_signal / 2, 0.99)
                    elif weighted_signal < -0.2:
                        signals['signal'] = 'sell'
                        signals['confidence'] = min(0.5 + abs(weighted_signal) / 2, 0.99)
                    else:
                        signals['signal'] = 'neutral'
                        signals['confidence'] = 0.5
            
            return signals
        except Exception as e:
            self.log(f"Error generating signals: {e}", level="ERROR")
            import traceback
            self.log(f"Error details: {traceback.format_exc()}", level="ERROR")
            return None

    # Add an alias method to handle the typo in case it's being called from elsewhere
    def run_analyeis(self, symbol='BTC/USD', timeframe='1h', limit=100):
        """Alias for run_analysis to handle typo in method name"""
        self.log("Warning: run_analyeis is deprecated, use run_analysis instead", level="WARNING")
        return self.run_analysis(symbol, timeframe, limit)
