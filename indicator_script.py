import pandas as pd
import numpy as np  # Import the entire numpy module
import requests
import json
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import io
import time
import pandas_ta as ta  # Use pandas_ta instead of talib
from api_utils import CryptoAPIUtils

class CryptoIndicators:
    def __init__(self, api_key="your_default_api_key_here", discord_logger=None):
        self.api_key = api_key
        self.data = None
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
            'fibonacci': 0.7,
            'parabolic_sar': 0.6
        }
        
        # Performance tracking for each indicator
        self.indicator_performance = {
            indicator: {
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.5,  # Start with neutral accuracy
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
            self.discord_logger.send_log(message, level)
            
    def fetch_data(self, symbol='BTC/USD', timeframe='1h', limit=200):
        """Fetch cryptocurrency data from a real API"""
        self.log(f"Fetching {limit} {timeframe} candles for {symbol}...")
        
        try:
            # Use our free API utility instead of requiring API keys
            price_data = self.api_utils.get_crypto_price_history(
                symbol=symbol,
                days=int(limit / 24) + 1,  # Convert limit to days
                interval='hourly' if timeframe in ['1h', '2h', '4h'] else 'daily'
            )
            
            if not price_data or not price_data.get('prices'):
                self.log("No data returned from free APIs, attempting fallback...", level="WARNING")
                return self._fetch_from_fallback(symbol, timeframe, limit)
            
            # Process the price data
            prices = price_data.get('prices', [])
            volumes = price_data.get('volumes', [])
            
            # Create DataFrame with timestamp and close price
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            
            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data if available
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                df = pd.merge(df, volume_df, on='timestamp', how='left')
            else:
                df['volume'] = 0
                
            # For OHLC data, approximate from close prices if needed
            if 'open' not in df.columns:
                df['open'] = df['close'].shift(1)
                df['high'] = df['close'] * 1.005  # Approximate
                df['low'] = df['close'] * 0.995   # Approximate
                
                # Fill NaN values in 'open' with 'close'
                df['open'] = df['open'].fillna(df['close'])
            
            # Keep only the most recent 'limit' records
            df = df.sort_values('timestamp', ascending=False).head(limit).sort_values('timestamp')
            
            # Set the data
            self.data = df
            
            self.log(f"Successfully fetched {len(df)} records from free API for {symbol}", send_to_discord=True)
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
    
    def _fetch_from_coingecko(self, symbol, timeframe, limit):
        """Fetch data from CoinGecko API"""
        # Convert timeframe to days for CoinGecko
        if timeframe.endswith('m'):
            days = max(1, (int(timeframe[:-1]) * limit) // (24 * 60))
        elif timeframe.endswith('h'):
            days = max(1, (int(timeframe[:-1]) * limit) // 24)
        else:  # assuming 'd' for days
            days = int(timeframe[:-1]) * limit
            
        # Cap at 90 days for CoinGecko free API
        days = min(days, 90)
        
        # Make API request to CoinGecko
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'hourly' if timeframe in ['1h', '2h', '4h'] else 'daily'
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            self.log(f"CoinGecko API error: {response.status_code} - {response.text}", level="ERROR")
            raise Exception(f"CoinGecko API returned status {response.status_code}")
            
        data = response.json()
        
        # CoinGecko returns prices, market_caps, and total_volumes as separate arrays
        # We need to combine them and convert to a DataFrame
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices:
            raise Exception("No price data returned from CoinGecko")
            
        # Create DataFrame with timestamp and close price
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Add volume data if available
        if volumes:
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            df = pd.merge(df, volume_df, on='timestamp', how='left')
        else:
            df['volume'] = 0
            
        # For OHLC data, we need to resample if necessary
        # For now, approximate OHLC from close prices
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] * 1.005  # Approximate
        df['low'] = df['close'] * 0.995   # Approximate
        
        # Fill NaN values in 'open' with 'close'
        df['open'] = df['open'].fillna(df['close'])
        
        # Keep only the most recent 'limit' records
        df = df.sort_values('timestamp', ascending=False).head(limit).sort_values('timestamp')
        
        # Set the data
        self.data = df
        
        self.log(f"Successfully fetched {len(df)} records from CoinGecko for {symbol}", send_to_discord=True)
        return True
        
    def _fetch_from_alternative_source(self, symbol, timeframe, limit):
        """Fetch data from an alternative source like Alpha Vantage or Binance"""
        # Try Alpha Vantage if API key is available
        if self.api_key and len(self.api_key) > 10:  # Basic check for a valid API key
            return self._fetch_from_alpha_vantage(symbol, timeframe, limit)
        else:
            # Try Binance public API as another alternative
            return self._fetch_from_binance(symbol, timeframe, limit)
    
    def _fetch_from_alpha_vantage(self, symbol, timeframe, limit):
        """Fetch data from Alpha Vantage API"""
        # Map timeframe to Alpha Vantage interval
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '60min', '4h': '4hour', '1d': 'daily'
        }
        
        interval = interval_map.get(timeframe, 'daily')
        
        # Extract the coin symbol
        if symbol.lower().endswith('usd'):
            coin = symbol[:-3]
        else:
            coin = symbol
            
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'CRYPTO_INTRADAY' if interval != 'daily' else 'DIGITAL_CURRENCY_DAILY',
            'symbol': coin,
            'market': 'USD',
            'interval': interval if interval != 'daily' else None,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Alpha Vantage API returned status {response.status_code}")
            
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            raise Exception(f"Alpha Vantage error: {data['Error Message']}")
            
        # Parse the time series data
        if interval == 'daily':
            time_series_key = 'Time Series (Digital Currency Daily)'
            price_key = '4a. close (USD)'
            vol_key = '5. volume'
        else:
            time_series_key = f'Time Series Crypto ({interval})'
            price_key = '4. close'
            vol_key = '5. volume'
            
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            raise Exception("No time series data returned from Alpha Vantage")
            
        # Convert to DataFrame
        records = []
        for date_str, values in time_series.items():
            records.append({
                'timestamp': pd.to_datetime(date_str),
                'open': float(values.get('1. open', values.get('1a. open (USD)', 0))),
                'high': float(values.get('2. high', values.get('2a. high (USD)', 0))),
                'low': float(values.get('3. low', values.get('3a. low (USD)', 0))),
                'close': float(values.get(price_key, 0)),
                'volume': float(values.get(vol_key, 0))
            })
            
        df = pd.DataFrame(records)
        
        # Sort and limit
        df = df.sort_values('timestamp', ascending=True).tail(limit)
        
        # Set the data
        self.data = df
        
        self.log(f"Successfully fetched {len(df)} records from Alpha Vantage for {symbol}", send_to_discord=True)
        return True
    
    def _fetch_from_binance(self, symbol, timeframe, limit):
        """Fetch data from Binance public API"""
        # Map timeframe to Binance interval
        interval_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w'
        }
        
        interval = interval_map.get(timeframe, '1d')
        
        # Format symbol for Binance
        if symbol.lower() == 'btc' or symbol.lower() == 'btcusd':
            binance_symbol = 'BTCUSDT'
        elif symbol.lower() == 'eth' or symbol.lower() == 'ethusd':
            binance_symbol = 'ETHUSDT'
        else:
            # Try to convert to Binance format
            binance_symbol = symbol.upper().replace('/', '') + 'USDT'
            
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': binance_symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Binance API returned status {response.status_code}: {response.text}")
            
        data = response.json()
        
        if not data:
            raise Exception("No data returned from Binance")
            
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Set the data
        self.data = df
        
        self.log(f"Successfully fetched {len(df)} records from Binance for {binance_symbol}", send_to_discord=True)
        return True
    
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
                raise Exception(f"Symbol {symbol} not supported by fallback source")
                
            url = f"https://query1.finance.yahoo.com/v7/finance/chart/{yahoo_symbol}"
            params = {
                'range': '3mo',  # 3 months
                'interval': '1d',  # daily data
                'indicators': 'quote',
                'includeTimestamps': 'true'
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise Exception(f"Yahoo Finance API returned status {response.status_code}")
                
            data = response.json()
            
            # Extract price data
            result = data.get('chart', {}).get('result', [])
            if not result:
                raise Exception("No data returned from Yahoo Finance")
                
            quotes = result[0]
            timestamps = quotes.get('timestamp', [])
            quote_data = quotes.get('indicators', {}).get('quote', [{}])[0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': quote_data.get('open', []),
                'high': quote_data.get('high', []),
                'low': quote_data.get('low', []),
                'close': quote_data.get('close', []),
                'volume': quote_data.get('volume', [])
            })
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Limit to requested number of rows
            df = df.tail(limit)
            
            # Set the data
            self.data = df
            
            self.log(f"Successfully fetched {len(df)} records from Yahoo Finance for {yahoo_symbol}", send_to_discord=True)
            return True
            
        except Exception as e:
            self.log(f"Fallback to Yahoo Finance failed: {e}", level="ERROR")
            
            # As absolute last resort, generate synthetic data with warning
            self.log("GENERATING SYNTHETIC DATA AS LAST RESORT - NOT SUITABLE FOR REAL TRADING", 
                     level="ERROR", send_to_discord=True)
            
            # Generate synthetic data based on real Bitcoin price patterns
            return self._generate_synthetic_data(symbol, timeframe, limit)
    
    def _generate_synthetic_data(self, symbol, timeframe, limit):
        """Generate synthetic data as a last resort"""
        # This is only for testing when no real data is available
        # Start with a realistic price for the symbol
        if 'BTC' in symbol.upper():
            start_price = 30000  # Example BTC price
        elif 'ETH' in symbol.upper():
            start_price = 2000   # Example ETH price
        else:
            start_price = 100    # Generic price

        # Generate timestamps
        end_time = datetime.now()
        
        # Determine the time delta based on the timeframe
        if timeframe.endswith('m'):
            delta = timedelta(minutes=int(timeframe[:-1]))
        elif timeframe.endswith('h'):
            delta = timedelta(hours=int(timeframe[:-1]))
        elif timeframe.endswith('d'):
            delta = timedelta(days=int(timeframe[:-1]))
        else:
            delta = timedelta(days=1)
            
        # Generate timestamps
        timestamps = [end_time - delta * i for i in range(limit)]
        timestamps.reverse()  # Oldest first
        
        # Generate price with random walk but realistic volatility
        volatility = 0.02  # 2% daily volatility
        trend = 0.0005     # Slight upward trend
        
        # Generate log returns with volatility clustering
        np.random.seed(42)  # For reproducibility
        log_returns = np.random.normal(trend, volatility, limit)
        
        # Add volatility clustering
        vol_cluster = np.random.normal(0, 0.5, limit)
        for i in range(1, limit):
            vol_cluster[i] = 0.7 * vol_cluster[i-1] + 0.3 * vol_cluster[i]
            log_returns[i] = log_returns[i] * (1 + 0.2 * vol_cluster[i])
            
        # Convert to prices
        log_prices = np.cumsum(log_returns)
        prices = start_price * np.exp(log_prices)
        
        # Generate OHLC
        opens = prices.copy()
        highs = opens * np.exp(np.random.uniform(0.001, 0.02, limit))
        lows = opens * np.exp(-np.random.uniform(0.001, 0.02, limit))
        closes = opens * np.exp(np.random.normal(0, 0.005, limit))
        
        # Ensure high is highest and low is lowest
        for i in range(limit):
            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])
            
        # Volume tends to be higher on big price moves
        volume_base = np.random.normal(1000, 200, limit)
        volume = volume_base * (1 + 5 * np.abs(log_returns))
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volume
        })
        
        self.log("Generated synthetic data. WARNING: Not suitable for real trading decisions.", 
                level="WARNING", send_to_discord=True)
        return True
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Using pandas_ta for RSI calculation
            self.data['rsi'] = ta.rsi(self.data['close'], length=period)
            
            # Also calculate RSI divergence
            self.data['rsi_slope'] = self.data['rsi'].diff(3) / 3  # 3-period ROC for RSI
            self.data['price_slope'] = self.data['close'].diff(3) / 3 / self.data['close'].shift(3)  # Normalized 3-period ROC for price
            
            # Handle NaN values in the slopes
            self.data['rsi_slope'] = self.data['rsi_slope'].fillna(0)
            self.data['price_slope'] = self.data['price_slope'].fillna(0)
            
            # Bullish and bearish divergence
            self.data['bullish_divergence'] = (self.data['rsi_slope'] > 0) & (self.data['price_slope'] < 0)
            self.data['bearish_divergence'] = (self.data['rsi_slope'] < 0) & (self.data['price_slope'] > 0)
            
            self.log("RSI calculated with divergence detection")
            return True
        except Exception as e:
            self.log(f"Error calculating RSI: {e}", level="ERROR")
            return False
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Using pandas_ta for MACD calculation
            macd_result = ta.macd(self.data['close'], fast=fast, slow=slow, signal=signal)
            
            # pandas_ta returns a DataFrame with MACD columns
            self.data['macd'] = macd_result[f'MACD_{fast}_{slow}_{signal}']
            self.data['macd_signal'] = macd_result[f'MACDs_{fast}_{slow}_{signal}']
            self.data['macd_hist'] = macd_result[f'MACDh_{fast}_{slow}_{signal}']
            
            # Calculate MACD histogram slope for momentum
            self.data['macd_hist_slope'] = self.data['macd_hist'].diff(3)
            
            # Detect MACD crosses
            self.data['macd_cross_up'] = (self.data['macd'] > self.data['macd_signal']) & (self.data['macd'].shift() <= self.data['macd_signal'].shift())
            self.data['macd_cross_down'] = (self.data['macd'] < self.data['macd_signal']) & (self.data['macd'].shift() >= self.data['macd_signal'].shift())
            
            self.log("MACD calculated with crossover detection")
            return True
        except Exception as e:
            self.log(f"Error calculating MACD: {e}", level="ERROR")
            return False
    
    def calculate_ema(self, periods=[20, 50, 200]):
        """Calculate Exponential Moving Averages and identify crosses"""
        if self.data is None:
            return False
        
        try:
            # Calculate EMAs for each period using pandas_ta
            for period in periods:
                self.data[f'ema_{period}'] = ta.ema(self.data['close'], length=period)
            
            # Identify EMA crosses (for trading signals)
            if len(periods) >= 2:
                periods.sort()  # Ensure periods are in ascending order
                
                # Check for golden cross (shorter EMA crosses above longer EMA)
                for i in range(len(periods) - 1):
                    short_period = periods[i]
                    long_period = periods[i + 1]
                    
                    cross_up_col = f'golden_cross_{short_period}_{long_period}'
                    cross_down_col = f'death_cross_{short_period}_{long_period}'
                    
                    # Golden cross: shorter EMA crosses above longer EMA
                    self.data[cross_up_col] = (
                        (self.data[f'ema_{short_period}'] > self.data[f'ema_{long_period}']) & 
                        (self.data[f'ema_{short_period}'].shift() <= self.data[f'ema_{long_period}'].shift())
                    )
                    
                    # Death cross: shorter EMA crosses below longer EMA
                    self.data[cross_down_col] = (
                        (self.data[f'ema_{short_period}'] < self.data[f'ema_{long_period}']) & 
                        (self.data[f'ema_{short_period}'].shift() >= self.data[f'ema_{long_period}'].shift())
                    )
            
            self.log(f"EMAs calculated for periods {periods} with crossover detection")
            return True
        except Exception as e:
            self.log(f"Error calculating EMAs: {e}", level="ERROR")
            return False
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Calculate Bollinger Bands with pandas_ta
            bbands = ta.bbands(self.data['close'], length=period, std=std_dev)
            
            # Map columns to our standard names
            self.data['bb_lower'] = bbands[f'BBL_{period}_{std_dev}.0']
            self.data['bb_middle'] = bbands[f'BBM_{period}_{std_dev}.0']
            self.data['bb_upper'] = bbands[f'BBU_{period}_{std_dev}.0']
            self.data['bb_bandwidth'] = bbands[f'BBB_{period}_{std_dev}.0']  # Bandwidth
            
            # Calculate %B manually (pandas_ta does provide it but let's be explicit)
            self.data['bb_percent_b'] = (self.data['close'] - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])
            
            # Identify potential breakouts and squeeze setups
            self.data['bb_squeeze'] = self.data['bb_bandwidth'] < self.data['bb_bandwidth'].rolling(window=50).min()
            self.data['bb_breakout_up'] = (self.data['close'] > self.data['bb_upper']) & (self.data['close'].shift() <= self.data['bb_upper'].shift())
            self.data['bb_breakout_down'] = (self.data['close'] < self.data['bb_lower']) & (self.data['close'].shift() >= self.data['bb_lower'].shift())
            
            self.log("Bollinger Bands calculated with breakout detection")
            return True
        except Exception as e:
            self.log(f"Error calculating Bollinger Bands: {e}", level="ERROR")
            return False
    
    def calculate_stochastic(self, k_period=14, d_period=3, slowing=3):
        """Calculate Stochastic Oscillator using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Using pandas_ta for stochastic calculation
            stoch = ta.stoch(self.data['high'], self.data['low'], self.data['close'], 
                             k=k_period, d=d_period, smooth_k=slowing)
            
            # Extract columns
            self.data['stoch_k'] = stoch[f'STOCHk_{k_period}_{d_period}_{slowing}']
            self.data['stoch_d'] = stoch[f'STOCHd_{k_period}_{d_period}_{slowing}']
            
            # Identify oversold/overbought conditions
            self.data['stoch_overbought'] = (self.data['stoch_k'] > 80) & (self.data['stoch_d'] > 80)
            self.data['stoch_oversold'] = (self.data['stoch_k'] < 20) & (self.data['stoch_d'] < 20)
            
            # Identify crossovers
            self.data['stoch_cross_up'] = (self.data['stoch_k'] > self.data['stoch_d']) & (self.data['stoch_k'].shift() <= self.data['stoch_d'].shift())
            self.data['stoch_cross_down'] = (self.data['stoch_k'] < self.data['stoch_d']) & (self.data['stoch_k'].shift() >= self.data['stoch_d'].shift())
            
            self.log("Stochastic oscillator calculated with crossover detection")
            return True
        except Exception as e:
            self.log(f"Error calculating Stochastic: {e}", level="ERROR")
            return False
    
    def calculate_ichimoku(self):
        """Calculate Ichimoku Cloud using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Calculate Ichimoku components using pandas_ta
            ichimoku = ta.ichimoku(self.data['high'], self.data['low'], self.data['close'])
            
            # Extract the components (column names may vary by pandas_ta version)
            # Map to our standard column names
            try:
                # For newer pandas_ta versions
                self.data['ichimoku_tenkan'] = ichimoku['ISA_9']  # Conversion Line
                self.data['ichimoku_kijun'] = ichimoku['ISB_26']  # Base Line
                self.data['ichimoku_senkou_a'] = ichimoku['ITS_9']  # Leading Span A
                self.data['ichimoku_senkou_b'] = ichimoku['ITS_26']  # Leading Span B
                self.data['ichimoku_chikou'] = ichimoku['ICS_26']  # Lagging Span
            except KeyError:
                # Alternative column names or manual calculation
                self.data['ichimoku_tenkan'] = ta.midprice(self.data['high'], self.data['low'], length=9)
                self.data['ichimoku_kijun'] = ta.midprice(self.data['high'], self.data['low'], length=26)
                self.data['ichimoku_senkou_a'] = ((self.data['ichimoku_tenkan'] + self.data['ichimoku_kijun']) / 2).shift(26)
                
                # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
                period52_high = self.data['high'].rolling(window=52).max()
                period52_low = self.data['low'].rolling(window=52).min()
                self.data['ichimoku_senkou_b'] = ((period52_high + period52_low) / 2).shift(26)
                
                # Chikou Span (Lagging Span): Close price shifted backwards 26 periods
                self.data['ichimoku_chikou'] = self.data['close'].shift(-26)
            
            # Identify key signals
            self.data['ichimoku_tk_cross_up'] = (self.data['ichimoku_tenkan'] > self.data['ichimoku_kijun']) & \
                                                (self.data['ichimoku_tenkan'].shift() <= self.data['ichimoku_kijun'].shift())
            
            self.data['ichimoku_tk_cross_down'] = (self.data['ichimoku_tenkan'] < self.data['ichimoku_kijun']) & \
                                                  (self.data['ichimoku_tenkan'].shift() >= self.data['ichimoku_kijun'].shift())
            
            # Price above/below cloud
            self.data['price_above_cloud'] = (self.data['close'] > self.data['ichimoku_senkou_a']) & \
                                             (self.data['close'] > self.data['ichimoku_senkou_b'])
            
            self.data['price_below_cloud'] = (self.data['close'] < self.data['ichimoku_senkou_a']) & \
                                             (self.data['close'] < self.data['ichimoku_senkou_b'])
            
            self.log("Ichimoku Cloud calculated with signal detection")
            return True
        except Exception as e:
            self.log(f"Error calculating Ichimoku: {e}", level="ERROR")
            return False
    
    def calculate_obv(self):
        """Calculate On-Balance Volume using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Use pandas_ta for OBV calculation
            self.data['obv'] = ta.obv(self.data['close'], self.data['volume'])
            
            # Calculate OBV EMA for signals
            self.data['obv_ema'] = ta.ema(self.data['obv'], length=20)
            
            # OBV divergence - handle potential division by zero or NaN values
            self.data['obv_slope'] = self.data['obv'].diff(3)
            # Normalize safely avoiding division by zero
            obv_shift = self.data['obv'].shift(3).abs()
            self.data['obv_slope'] = np.where(obv_shift > 0, 
                                             self.data['obv_slope'] / obv_shift * 100,
                                             0)
            
            # Do the same for price slope
            self.data['price_slope'] = self.data['close'].diff(3)
            price_shift = self.data['close'].shift(3)
            self.data['price_slope'] = np.where(price_shift > 0,
                                              self.data['price_slope'] / price_shift * 100,
                                              0)
            
            # Fill any remaining NaN values
            self.data['obv_slope'] = self.data['obv_slope'].fillna(0)
            self.data['price_slope'] = self.data['price_slope'].fillna(0)
            
            self.data['obv_bull_div'] = (self.data['obv_slope'] > 0) & (self.data['price_slope'] < 0)
            self.data['obv_bear_div'] = (self.data['obv_slope'] < 0) & (self.data['price_slope'] > 0)
            
            self.log("On-Balance Volume calculated with divergence detection")
            return True
        except Exception as e:
            self.log(f"Error calculating OBV: {e}", level="ERROR")
            return False
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range for volatility using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Using pandas_ta for ATR calculation
            self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=period)
            
            # Calculate normalized ATR (ATR as a percentage of price)
            self.data['atr_percent'] = (self.data['atr'] / self.data['close']) * 100
            
            # Identify volatility expansion
            self.data['vol_expansion'] = self.data['atr_percent'] > self.data['atr_percent'].rolling(window=period*2).mean() * 1.5
            
            self.log("ATR calculated with volatility expansion detection")
            return True
        except Exception as e:
            self.log(f"Error calculating ATR: {e}", level="ERROR")
            return False
    
    def calculate_parabolic_sar(self, acceleration=0.02, maximum=0.2):
        """Calculate Parabolic SAR for trend following using pandas_ta"""
        if self.data is None:
            return False
        
        try:
            # Using pandas_ta for PSAR calculation
            psar = ta.psar(self.data['high'], self.data['low'], af=acceleration, max_af=maximum)
            
            # Extract the relevant series (column name may vary by pandas_ta version)
            if 'PSARl_0.02_0.2' in psar.columns:
                self.data['sar'] = psar['PSARl_0.02_0.2'].combine_first(psar['PSARs_0.02_0.2'])
            else:
                # For older pandas_ta versions or different column naming
                self.data['sar'] = psar.iloc[:, 0]  # Take the first column
            
            # Generate signals for trend changes
            self.data['sar_bullish'] = self.data['close'] > self.data['sar']
            self.data['sar_bearish'] = self.data['close'] < self.data['sar']
            
            # Detect SAR flips (trend reversals)
            self.data['sar_bull_flip'] = (self.data['sar_bullish']) & (~self.data['sar_bullish'].shift(1))
            self.data['sar_bear_flip'] = (self.data['sar_bearish']) & (~self.data['sar_bearish'].shift(1))
            
            self.log("Parabolic SAR calculated with trend detection")
            return True
        except Exception as e:
            self.log(f"Error calculating Parabolic SAR: {e}", level="ERROR")
            return False
    
    def analyze_volume(self):
        """Analyze volume patterns"""
        if self.data is None:
            return False
        
        try:
            # Calculate volume change percentage
            self.data['volume_pct_change'] = self.data['volume'].pct_change() * 100
            
            # Calculate volume moving averages
            self.data['volume_sma_10'] = ta.sma(self.data['volume'], length=10)
            self.data['volume_sma_20'] = ta.sma(self.data['volume'], length=20)
            
            # Flag different volume spike levels
            self.data['volume_spike_weak'] = self.data['volume'] > self.data['volume_sma_20'] * 1.5
            self.data['volume_spike_strong'] = self.data['volume'] > self.data['volume_sma_20'] * 2.5
            self.data['volume_spike_extreme'] = self.data['volume'] > self.data['volume_sma_20'] * 4.0
            
            # Identify climax volume (high volume after a trend)
            price_trend = self.data['close'].diff(5).rolling(window=5).sum()
            self.data['climax_volume_up'] = (price_trend > 0) & self.data['volume_spike_strong']
            self.data['climax_volume_down'] = (price_trend < 0) & self.data['volume_spike_strong']
            
            # Volume price confirmation
            self.data['vol_price_up_confirm'] = (self.data['close'] > self.data['close'].shift(1)) & (self.data['volume'] > self.data['volume'].shift(1))
            self.data['vol_price_down_confirm'] = (self.data['close'] < self.data['close'].shift(1)) & (self.data['volume'] > self.data['volume'].shift(1))
            
            self.log("Volume analysis completed with pattern detection")
            return True
        except Exception as e:
            self.log(f"Error analyzing volume: {e}", level="ERROR")
            return False
    
    def generate_signals(self):
        """Generate trading signals based on indicators"""
        if self.data is None:
            return False
        
        self.log("Generating trading signals from indicators...")
        
        try:
            # Get the latest data point with NaN handling
            latest = self.data.iloc[-1].copy()
            prev = self.data.iloc[-2].copy() if len(self.data) > 1 else latest.copy()
            
            # Store all signals
            signals = {}
            
            # Helper function to safely get values that might be NaN
            def safe_get(row, column, default=0):
                """Safely get value from dataframe, handling NaN"""
                if column in row and not pd.isna(row[column]):
                    return row[column]
                return default
            
            # RSI signal (oversold/overbought)
            rsi_signal = 0
            rsi_value = safe_get(latest, 'rsi')
            if rsi_value < 30:
                rsi_signal = 1  # Oversold - buy signal
            elif rsi_value > 70:
                rsi_signal = -1  # Overbought - sell signal
            
            # Check for RSI divergence (stronger signal)
            if safe_get(latest, 'bullish_divergence', False):
                rsi_signal = max(rsi_signal, 0.5)
            if safe_get(latest, 'bearish_divergence', False):
                rsi_signal = min(rsi_signal, -0.5)
            
            # MACD signal
            macd_signal = 0
            if safe_get(latest, 'macd_cross_up', False):
                macd_signal = 1
            elif safe_get(latest, 'macd_cross_down', False):
                macd_signal = -1
            elif 'macd' in latest and 'macd_signal' in latest:
                if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                    if latest['macd'] > latest['macd_signal']:
                        macd_signal = 0.5
                    elif latest['macd'] < latest['macd_signal']:
                        macd_signal = -0.5
            
            # EMA signal
            ema_signal = 0
            
            # Check for golden/death crosses between EMAs (stronger signals)
            if latest.get('golden_cross_20_50', False):
                ema_signal = 1
            elif latest.get('death_cross_20_50', False):
                ema_signal = -1
            # Regular price vs EMA check
            elif latest['close'] > latest['ema_50']:
                ema_signal = 0.5  # Price above EMA, bullish
            elif latest['close'] < latest['ema_50']:
                ema_signal = -0.5  # Price below EMA, bearish
            
            # Bollinger Bands signal
            bollinger_signal = 0
            if hasattr(latest, 'bb_breakout_up') and latest['bb_breakout_up']:
                bollinger_signal = 1  # Bullish breakout
            elif hasattr(latest, 'bb_breakout_down') and latest['bb_breakout_down']:
                bollinger_signal = -1  # Bearish breakout
            elif hasattr(latest, 'bb_percent_b'):
                # Detect mean reversion opportunities
                if latest['bb_percent_b'] < 0:
                    bollinger_signal = 0.5  # Price below lower band (potential buy)
                elif latest['bb_percent_b'] > 1:
                    bollinger_signal = -0.5  # Price above upper band (potential sell)
            
            # Stochastic signal
            stochastic_signal = 0
            if hasattr(latest, 'stoch_oversold') and latest['stoch_oversold'] and latest.get('stoch_cross_up', False):
                stochastic_signal = 1  # Strong buy signal
            elif hasattr(latest, 'stoch_overbought') and latest['stoch_overbought'] and latest.get('stoch_cross_down', False):
                stochastic_signal = -1  # Strong sell signal
            elif hasattr(latest, 'stoch_cross_up') and latest['stoch_cross_up']:
                stochastic_signal = 0.5  # Bullish crossover
            elif hasattr(latest, 'stoch_cross_down') and latest['stoch_cross_down']:
                stochastic_signal = -0.5  # Bearish crossover
            
            # Ichimoku signal
            ichimoku_signal = 0
            if hasattr(latest, 'price_above_cloud') and latest['price_above_cloud']:
                ichimoku_signal = 0.5  # Bullish bias
                if hasattr(latest, 'ichimoku_tk_cross_up') and latest['ichimoku_tk_cross_up']:
                    ichimoku_signal = 1  # Strong buy
            elif hasattr(latest, 'price_below_cloud') and latest['price_below_cloud']:
                ichimoku_signal = -0.5  # Bearish bias
                if hasattr(latest, 'ichimoku_tk_cross_down') and latest['ichimoku_tk_cross_down']:
                    ichimoku_signal = -1  # Strong sell
            
            # Volume signal
            volume_signal = 0
            if hasattr(latest, 'vol_price_up_confirm') and latest['vol_price_up_confirm']:
                volume_signal = 0.5  # Bullish volume confirmation
                if hasattr(latest, 'climax_volume_up') and latest['climax_volume_up']:
                    volume_signal = 1  # Potential buying climax
            elif hasattr(latest, 'vol_price_down_confirm') and latest['vol_price_down_confirm']:
                volume_signal = -0.5  # Bearish volume confirmation
                if hasattr(latest, 'climax_volume_down') and latest['climax_volume_down']:
                    volume_signal = -1  # Potential selling climax
            
            # OBV signal
            obv_signal = 0
            if hasattr(latest, 'obv') and hasattr(latest, 'obv_ema'):
                if latest['obv'] > latest['obv_ema']:
                    obv_signal = 0.5  # Bullish volume trend
                elif latest['obv'] < latest['obv_ema']:
                    obv_signal = -0.5  # Bearish volume trend
                
                # Check for divergence
                if hasattr(latest, 'obv_bull_div') and latest['obv_bull_div']:
                    obv_signal = max(obv_signal, 0.8)  # Bullish divergence
                if hasattr(latest, 'obv_bear_div') and latest['obv_bear_div']:
                    obv_signal = min(obv_signal, -0.8)  # Bearish divergence
            
            # Parabolic SAR signal
            parabolic_signal = 0
            if hasattr(latest, 'sar_bull_flip') and latest['sar_bull_flip']:
                parabolic_signal = 1  # Strong buy signal on trend change
            elif hasattr(latest, 'sar_bear_flip') and latest['sar_bear_flip']:
                parabolic_signal = -1  # Strong sell signal on trend change
            elif hasattr(latest, 'sar_bullish') and latest['sar_bullish']:
                parabolic_signal = 0.5  # Bullish trend
            elif hasattr(latest, 'sar_bearish') and latest['sar_bearish']:
                parabolic_signal = -0.5  # Bearish trend
            
            # Combined ATR for measuring volatility impact
            atr_factor = latest.get('atr_percent', 1) / 2  # Normalize to a reasonable scale
            
            # Collect all signals
            signals = {
                'rsi': rsi_signal,
                'macd': macd_signal,
                'ema': safe_get(locals(), 'ema_signal', 0),
                'bollinger': safe_get(locals(), 'bollinger_signal', 0),
                'stochastic': safe_get(locals(), 'stochastic_signal', 0),
                'ichimoku': safe_get(locals(), 'ichimoku_signal', 0),
                'volume': safe_get(locals(), 'volume_signal', 0),
                'obv': safe_get(locals(), 'obv_signal', 0),
                'parabolic_sar': safe_get(locals(), 'parabolic_signal', 0)
            }
            
            # Calculate weighted signal based on performance-adjusted weights
            weighted_signal = 0
            weight_sum = 0
            
            for indicator, signal in signals.items():
                if indicator in self.indicator_weights:
                    # Adjust weight by historical accuracy
                    accuracy = self.indicator_performance.get(indicator, {}).get('accuracy', 0.5)
                    adjusted_weight = self.indicator_weights[indicator] * (0.5 + accuracy)
                    
                    weighted_signal += signal * adjusted_weight
                    weight_sum += adjusted_weight
            
            # Normalize
            if weight_sum > 0:
                final_signal_value = weighted_signal / weight_sum
            else:
                final_signal_value = 0
            
            # Determine final signal direction
            if final_signal_value > 0.2:
                signal_direction = 'buy'
            elif final_signal_value < -0.2:
                signal_direction = 'sell'
            else:
                signal_direction = 'neutral'
            
            # Calculate confidence (absolute value of combined, normalized)
            confidence = min(abs(final_signal_value) * 1.5, 1.0)
            
            # Check for conflicting signals
            conflicting_signals = []
            for indicator, signal in signals.items():
                # If a strong signal opposes the overall direction
                if (signal_direction == 'buy' and signal < -0.5) or (signal_direction == 'sell' and signal > 0.5):
                    conflicting_signals.append(indicator)
            
            conflicting = len(conflicting_signals) > 0
            
            # Adjust confidence if there are conflicting signals
            if conflicting:
                confidence = max(0.1, confidence - 0.2 * len(conflicting_signals))
            
            # Log which indicators contributed most to the decision
            top_bull_indicators = sorted([(i, s) for i, s in signals.items() if s > 0], key=lambda x: x[1], reverse=True)[:3]
            top_bear_indicators = sorted([(i, s) for i, s in signals.items() if s < 0], key=lambda x: x[1])[:3]
            
            bull_indicator_msg = ", ".join([f"{i} ({s:.2f})" for i, s in top_bull_indicators]) if top_bull_indicators else "None"
            bear_indicator_msg = ", ".join([f"{i} ({s:.2f})" for i, s in top_bear_indicators]) if top_bear_indicators else "None"
            
            # Check for extreme signals (potential reversal)
            extreme_signal = abs(final_signal_value) > 0.8
            
            # Store the final signals dictionary
            self.signals = {
                'timestamp': datetime.now().isoformat(),
                'price': float(safe_get(latest, 'close', 0)),
                'signal': safe_get(locals(), 'signal_direction', 'neutral'),
                'raw_signal_value': float(safe_get(locals(), 'final_signal_value', 0)),
                'confidence': float(safe_get(locals(), 'confidence', 0.5)),
                'indicators': {
                    'rsi': float(safe_get(latest, 'rsi', 0)),
                    'rsi_signal': float(rsi_signal),
                    'macd': float(safe_get(latest, 'macd', 0)),
                    'macd_signal': float(macd_signal),
                    'ema': float(safe_get(latest, 'ema_50', 0)),
                    'ema_signal': float(safe_get(locals(), 'ema_signal', 0)),
                    'bollinger_signal': float(safe_get(locals(), 'bollinger_signal', 0)),
                    'stochastic_signal': float(safe_get(locals(), 'stochastic_signal', 0)),
                    'ichimoku_signal': float(safe_get(locals(), 'ichimoku_signal', 0)),
                    'volume': float(safe_get(latest, 'volume', 0)),
                    'volume_signal': float(safe_get(locals(), 'volume_signal', 0)),
                    'obv_signal': float(safe_get(locals(), 'obv_signal', 0)),
                    'parabolic_sar_signal': float(safe_get(locals(), 'parabolic_signal', 0)),
                    'atr': float(latest.get('atr', 0)),
                    'atr_percent': float(latest.get('atr_percent', 0))
                },
                'top_bullish_indicators': [i for i, s in top_bull_indicators],
                'top_bearish_indicators': [i for i, s in top_bear_indicators],
                'conflicting_signals': conflicting,
                'conflicting_indicators': conflicting_signals,
                'extreme_signal': extreme_signal,
                'volatility': float(atr_factor)
            }
            
            self.log(f"Generated {signal_direction.upper()} signal with {confidence:.1%} confidence", send_to_discord=True)
            self.log(f"Top bullish indicators: {bull_indicator_msg}")
            self.log(f"Top bearish indicators: {bear_indicator_msg}")
            
            if conflicting:
                self.log(f"Warning: Conflicting signals from {', '.join(conflicting_signals)}", level="WARNING", send_to_discord=True)
            
            return self.signals
            
        except Exception as e:
            self.log(f"Error generating signals: {e}", level="ERROR", send_to_discord=True)
            return False
    
    def update_indicator_performance(self, actual_outcome):
        """Update indicator accuracy based on actual price movement"""
        if not self.signals:
            return False
        
        self.log(f"Updating indicator performance with actual outcome: {actual_outcome}")
        
        try:
            # Get the signals from last prediction
            indicator_signals = self.signals.get('indicators', {})
            signal_direction = self.signals.get('signal', 'neutral')
            
            # Convert signal direction to numeric
            expected_direction = 1 if signal_direction == 'buy' else (-1 if signal_direction == 'sell' else 0)
            
            # Convert actual outcome to direction
            actual_direction = 1 if actual_outcome == 'buy' else (-1 if actual_outcome == 'sell' else 0)
            
            # Update performance for each indicator
            for indicator, signal_key in {
                'rsi': 'rsi_signal',
                'macd': 'macd_signal',
                'ema': 'ema_signal',
                'bollinger': 'bollinger_signal',
                'stochastic': 'stochastic_signal',
                'ichimoku': 'ichimoku_signal',
                'volume': 'volume_signal',
                'obv': 'obv_signal',
                'parabolic_sar': 'parabolic_sar_signal'
            }.items():
                if signal_key in indicator_signals:
                    indicator_signal = indicator_signals[signal_key]
                    
                    # Skip if signal is too weak
                    if abs(indicator_signal) < 0.3:
                        continue
                    
                    # Determine indicator direction
                    indicator_direction = 1 if indicator_signal > 0 else (-1 if indicator_signal < 0 else 0)
                    
                    # Check if indicator was correct
                    was_correct = (indicator_direction == actual_direction)
                    
                    # Update indicator performance
                    if indicator in self.indicator_performance:
                        if was_correct:
                            self.indicator_performance[indicator]['correct'] += 1
                        else:
                            self.indicator_performance[indicator]['incorrect'] += 1
                        
                        # Calculate new accuracy
                        correct = self.indicator_performance[indicator]['correct']
                        incorrect = self.indicator_performance[indicator]['incorrect']
                        total = correct + incorrect
                        
                        if total > 0:
                            self.indicator_performance[indicator]['accuracy'] = correct / total
                        
                        # Update timestamp
                        self.indicator_performance[indicator]['last_updated'] = datetime.now().isoformat()
            
            # Save the updated performance
            self._save_performance()
            
            # Log overall performance
            self.log("Updated indicator performance", send_to_discord=True)
            for indicator, perf in self.indicator_performance.items():
                accuracy = perf.get('accuracy', 0)
                self.log(f"  {indicator}: {accuracy:.1%} accuracy")
            
            return True
            
        except Exception as e:
            self.log(f"Error updating indicator performance: {e}", level="ERROR")
            return False
    
    def _save_performance(self):
        """Save indicator performance to file"""
        try:
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            # Save to file
            with open('data/indicator_performance.json', 'w') as f:
                json.dump(self.indicator_performance, f, indent=2)
                
            return True
        except Exception as e:
            self.log(f"Error saving indicator performance: {e}", level="ERROR")
            return False
    
    def _load_performance(self):
        """Load indicator performance from file"""
        try:
            if os.path.exists('data/indicator_performance.json'):
                with open('data/indicator_performance.json', 'r') as f:
                    loaded_performance = json.load(f)
                    # Update the indicators that exist in the loaded file
                    for indicator, perf in loaded_performance.items():
                        if indicator in self.indicator_performance:
                            self.indicator_performance[indicator] = perf
                    
                    self.log(f"Loaded indicator performance data")
                    return True
            return False
        except Exception as e:
            self.log(f"Error loading indicator performance: {e}", level="ERROR")
            return False
    
    def save_signals(self, filepath=None):
        """Save signals to JSON file"""
        if not filepath:
            # Create directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            filepath = f"data/signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.signals, f, indent=2)
        
        self.log(f"Saved signals to {filepath}")
        return filepath
    
    def generate_charts(self):
        """Generate charts for technical analysis"""
        if self.data is None or len(self.data) < 20:
            self.log("Not enough data to generate charts", level="WARNING")
            return None
        
        self.log("Generating technical analysis charts...")
        
        try:
            # Create a multi-panel figure
            fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            
            # Get recent data for charting (last 100 points)
            chart_data = self.data.iloc[-100:].copy() if len(self.data) > 100 else self.data.copy()
            
            # Format dates for x-axis
            dates = [d.strftime('%m-%d %H:%M') if hasattr(d, 'strftime') else str(d) 
                    for d in chart_data['timestamp']]
            
            # Panel 1: Price chart with EMAs and Bollinger Bands
            axs[0].set_title('Price with EMAs and Bollinger Bands')
            
            # Plot candlesticks
            for i in range(len(chart_data)):
                date_pos = i
                op, hi, lo, cl = chart_data['open'].iloc[i], chart_data['high'].iloc[i], \
                                chart_data['low'].iloc[i], chart_data['close'].iloc[i]
                
                # Body
                body_color = 'green' if cl >= op else 'red'
                body_bottom = min(op, cl)
                body_top = max(op, cl)
                body_height = body_top - body_bottom
                
                axs[0].plot([date_pos, date_pos], [lo, hi], color='black', linewidth=1)
                axs[0].bar(date_pos, body_height, bottom=body_bottom, color=body_color, width=0.8)
            
            # Plot EMAs
            for period in [20, 50, 200]:
                if f'ema_{period}' in chart_data.columns:
                    axs[0].plot(chart_data[f'ema_{period}'], label=f'EMA {period}')
            
            # Plot Bollinger Bands
            if 'bb_upper' in chart_data.columns and 'bb_lower' in chart_data.columns:
                axs[0].plot(chart_data['bb_upper'], 'k--', alpha=0.5, label='BB Upper')
                axs[0].plot(chart_data['bb_middle'], 'k-', alpha=0.5, label='BB Middle')
                axs[0].plot(chart_data['bb_lower'], 'k--', alpha=0.5, label='BB Lower')
                
                # Fill between bands
                x_range = range(len(chart_data))
                axs[0].fill_between(x_range, chart_data['bb_lower'], chart_data['bb_upper'], 
                                  color='gray', alpha=0.1)
            
            # Highlight current signal
            if self.signals:
                signal_color = 'green' if self.signals['signal'] == 'buy' else ('red' if self.signals['signal'] == 'sell' else 'blue')
                axs[0].axhline(y=self.signals['price'], color=signal_color, linestyle='-', alpha=0.5)
                axs[0].text(len(chart_data)-1, self.signals['price'], 
                          f" {self.signals['signal'].upper()} ({self.signals['confidence']:.1%})", 
                          color=signal_color, fontweight='bold')
            
            axs[0].set_ylabel('Price')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend(loc='upper left')
            
            # Panel 2: RSI
            axs[1].set_title('RSI')
            if 'rsi' in chart_data.columns:
                axs[1].plot(chart_data['rsi'], color='purple', label='RSI')
                axs[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
                axs[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
                axs[1].set_ylim(0, 100)
                axs[1].fill_between(range(len(chart_data)), 70, 100, color='red', alpha=0.1)
                axs[1].fill_between(range(len(chart_data)), 0, 30, color='green', alpha=0.1)
            axs[1].set_ylabel('RSI')
            axs[1].grid(True, alpha=0.3)
            
            # Panel 3: MACD
            axs[2].set_title('MACD')
            if all(x in chart_data.columns for x in ['macd', 'macd_signal', 'macd_hist']):
                axs[2].plot(chart_data['macd'], color='blue', label='MACD')
                axs[2].plot(chart_data['macd_signal'], color='red', label='Signal')
                axs[2].bar(range(len(chart_data)), chart_data['macd_hist'], color=chart_data['macd_hist'].apply(
                    lambda x: 'green' if x > 0 else 'red'), label='Histogram', alpha=0.5)
                axs[2].axhline(y=0, color='black', linestyle='-', alpha=0.2)
            axs[2].set_ylabel('MACD')
            axs[2].grid(True, alpha=0.3)
            axs[2].legend(loc='upper left')
            
            # Panel 4: Volume with Volume MA
            axs[3].set_title('Volume')
            axs[3].bar(range(len(chart_data)), chart_data['volume'], 
                     color=chart_data.apply(lambda x: 'green' if x['close'] > x['open'] else 'red', axis=1),
                     alpha=0.6)
            
            if 'volume_sma_20' in chart_data.columns:
                axs[3].plot(chart_data['volume_sma_20'], color='blue', label='Volume MA(20)')
            
            axs[3].set_ylabel('Volume')
            axs[3].grid(True, alpha=0.3)
            axs[3].legend(loc='upper left')
            
            # Set x-axis labels on bottom plot only (last panel)
            for ax in axs[:-1]:
                ax.set_xticklabels([])
            
            # Add every 10th date label to avoid crowding
            xticks = range(0, len(dates), 10)
            axs[-1].set_xticks(xticks)
            axs[-1].set_xticklabels([dates[i] for i in xticks], rotation=45)
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Close the figure to free memory
            plt.close(fig)
            
            self.log("Generated technical analysis charts")
            return buf
            
        except Exception as e:
            self.log(f"Error generating charts: {e}", level="ERROR", send_to_discord=True)
            return None
    
    def run_analysis(self, symbol='BTC/USD', timeframe='1h', limit=100):
        """Run the complete indicator analysis pipeline"""
        self.log(f"Starting full technical analysis for {symbol} ({timeframe})", send_to_discord=True)
        start_time = time.time()
        
        if not self.fetch_data(symbol, timeframe, limit):
            self.log("Failed to fetch data", level="ERROR", send_to_discord=True)
            return False
        
        # Calculate all indicators
        try:
            # Essential indicators
            self.calculate_rsi()
            self.calculate_macd()
            self.calculate_ema()
            self.calculate_atr()
            self.analyze_volume()
            
            # Additional indicators
            self.calculate_bollinger_bands()
            self.calculate_stochastic()
            self.calculate_ichimoku()
            self.calculate_obv()
            self.calculate_parabolic_sar()
            
            # Generate signals
            signals = self.generate_signals()
            
            # Save signals
            self.save_signals()
            
            end_time = time.time()
            self.log(f"Analysis completed in {end_time - start_time:.2f} seconds", send_to_discord=True)
            
            return signals
            
        except Exception as e:
            self.log(f"Error in analysis pipeline: {e}", level="ERROR", send_to_discord=True)
            return False
