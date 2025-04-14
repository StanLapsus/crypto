import requests
import time
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class CryptoAPIUtils:
    """Utility class for accessing free crypto APIs without keys"""
    
    def __init__(self, cache_dir='data/api_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Track API calls to respect rate limits
        self.last_call_times = {
            'coingecko': datetime.now() - timedelta(minutes=1),
            'coincap': datetime.now() - timedelta(minutes=1),
            'binance': datetime.now() - timedelta(minutes=1)
        }
    
    def _respect_rate_limit(self, api_name, min_interval_seconds=1.2):
        """Ensure we don't exceed rate limits"""
        if api_name in self.last_call_times:
            elapsed = (datetime.now() - self.last_call_times[api_name]).total_seconds()
            if elapsed < min_interval_seconds:
                sleep_time = min_interval_seconds - elapsed
                time.sleep(sleep_time)
        
        # Update last call time
        self.last_call_times[api_name] = datetime.now()
    
    def _try_from_cache(self, cache_key, max_age_hours=24):
        """Try to load data from cache if recent enough"""
        cache_file = f"{self.cache_dir}/{cache_key}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check cache age
                timestamp = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01T00:00:00'))
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                
                if age_hours <= max_age_hours:
                    return cached_data.get('data')
            except Exception:
                pass
        
        return None
    
    def _save_to_cache(self, cache_key, data):
        """Save API data to cache"""
        cache_file = f"{self.cache_dir}/{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f)
            return True
        except Exception:
            return False
    
    def get_crypto_price_history(self, symbol, days=7, interval='hourly'):
        """Get price history using free APIs (CoinGecko, CoinCap, Binance)"""
        # Clean symbol format
        clean_symbol = symbol.lower().replace('/', '').replace('-', '')
        if clean_symbol.endswith('usd'):
            coin_id = clean_symbol[:-3]
        else:
            coin_id = clean_symbol
        
        # Try from cache first
        cache_key = f"price_history_{coin_id}_{days}_{interval}"
        cached_data = self._try_from_cache(cache_key)
        if cached_data:
            print(f"Using cached price data for {symbol}")
            return cached_data
        
        # Try CoinGecko first
        try:
            self._respect_rate_limit('coingecko')
            
            # Map common symbols to CoinGecko IDs
            if coin_id == 'btc':
                coin_id = 'bitcoin'
            elif coin_id == 'eth':
                coin_id = 'ethereum'
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': interval
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract price data
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                
                if prices:
                    result = {
                        'source': 'coingecko',
                        'prices': prices,
                        'volumes': volumes,
                        'symbol': symbol
                    }
                    self._save_to_cache(cache_key, result)
                    return result
        except Exception as e:
            print(f"CoinGecko API error: {e}")
        
        # Try CoinCap as fallback
        try:
            self._respect_rate_limit('coincap')
            
            # Map to CoinCap ID format
            if coin_id == 'bitcoin':
                coincap_id = 'bitcoin'
            elif coin_id == 'ethereum':
                coincap_id = 'ethereum'
            else:
                coincap_id = coin_id
            
            # CoinCap uses different endpoints for historical data
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            url = f"https://api.coincap.io/v2/assets/{coincap_id}/history"
            params = {
                'interval': 'h1' if interval == 'hourly' else 'd1',
                'start': start_time,
                'end': end_time
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract and format price data to match CoinGecko format
                history = data.get('data', [])
                prices = [[int(item['time']), float(item['priceUsd'])] for item in history]
                volumes = [[int(item['time']), float(item.get('volumeUsd', 0) or 0)] for item in history]
                
                if prices:
                    result = {
                        'source': 'coincap',
                        'prices': prices,
                        'volumes': volumes,
                        'symbol': symbol
                    }
                    self._save_to_cache(cache_key, result)
                    return result
        except Exception as e:
            print(f"CoinCap API error: {e}")
        
        # Try Binance as last resort
        try:
            self._respect_rate_limit('binance')
            
            # Convert to Binance format
            if clean_symbol.endswith('usd'):
                binance_symbol = f"{clean_symbol[:-3].upper()}USDT"
            else:
                binance_symbol = f"{clean_symbol.upper()}USDT"
            
            # Map interval to Binance format
            if interval == 'hourly':
                kline_interval = '1h'
            else:
                kline_interval = '1d'
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': kline_interval,
                'limit': min(1000, days * (24 if kline_interval == '1h' else 1))
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Format Binance data to match CoinGecko format
                prices = [[item[0], float(item[4])] for item in data]  # timestamp and close price
                volumes = [[item[0], float(item[5])] for item in data]  # timestamp and volume
                
                if prices:
                    result = {
                        'source': 'binance',
                        'prices': prices,
                        'volumes': volumes,
                        'symbol': symbol
                    }
                    self._save_to_cache(cache_key, result)
                    return result
        except Exception as e:
            print(f"Binance API error: {e}")
        
        # If all APIs fail, return None
        return None
    
    def get_current_price(self, symbol):
        """Get current price for a cryptocurrency"""
        clean_symbol = symbol.lower().replace('/', '').replace('-', '')
        
        # Try CoinGecko
        try:
            self._respect_rate_limit('coingecko')
            
            # Map common symbols
            if clean_symbol.startswith('btc'):
                coin_id = 'bitcoin'
            elif clean_symbol.startswith('eth'):
                coin_id = 'ethereum'
            else:
                coin_id = clean_symbol
                if coin_id.endswith('usd'):
                    coin_id = coin_id[:-3]
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if coin_id in data:
                    price_data = data[coin_id]
                    return {
                        'price': price_data.get('usd', 0),
                        'volume_24h': price_data.get('usd_24h_vol', 0),
                        'change_24h': price_data.get('usd_24h_change', 0),
                        'source': 'coingecko'
                    }
        except Exception as e:
            print(f"CoinGecko current price error: {e}")
        
        # Try CoinCap
        try:
            self._respect_rate_limit('coincap')
            
            if clean_symbol.endswith('usd'):
                asset_id = clean_symbol[:-3]
            else:
                asset_id = clean_symbol
                
            url = f"https://api.coincap.io/v2/assets/{asset_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                asset_data = data.get('data', {})
                
                if asset_data:
                    return {
                        'price': float(asset_data.get('priceUsd', 0)),
                        'volume_24h': float(asset_data.get('volumeUsd24Hr', 0)),
                        'change_24h': float(asset_data.get('changePercent24Hr', 0)),
                        'source': 'coincap'
                    }
        except Exception as e:
            print(f"CoinCap current price error: {e}")
        
        # If all fail, return None
        return None
    
    def get_crypto_news(self, keywords="bitcoin crypto", max_items=10):
        """Get crypto news headlines without using a paid API key"""
        # Try from cache first to avoid excessive calls
        cache_key = f"news_{keywords.replace(' ', '_')}_{max_items}"
        cached_data = self._try_from_cache(cache_key, max_age_hours=6)  # News cache is shorter
        if cached_data:
            return cached_data
        
        # CryptoPanic offers a free API for crypto news
        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': '',  # No auth token needed for public data
                'currencies': 'BTC',
                'kind': 'news',
                'public': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if results:
                    # Format similar to a traditional news API
                    articles = []
                    for item in results[:max_items]:
                        articles.append({
                            'title': item.get('title', ''),
                            'description': item.get('title', ''),  # Use title as description
                            'url': item.get('url', ''),
                            'source': item.get('source', {}).get('title', 'CryptoPanic'),
                            'published_at': item.get('published_at', ''),
                            'sentiment': item.get('votes', {}).get('positive', 0) - item.get('votes', {}).get('negative', 0)
                        })
                    
                    news_data = {
                        'articles': articles,
                        'source': 'cryptopanic',
                        'count': len(articles)
                    }
                    
                    self._save_to_cache(cache_key, news_data)
                    return news_data
        except Exception as e:
            print(f"CryptoPanic API error: {e}")
        
        # Try Reddit API as fallback
        try:
            url = "https://www.reddit.com/r/CryptoCurrency/hot.json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                
                if posts:
                    articles = []
                    for post in posts[:max_items]:
                        post_data = post.get('data', {})
                        if not post_data.get('stickied', False):  # Skip stickied posts
                            articles.append({
                                'title': post_data.get('title', ''),
                                'description': post_data.get('selftext', '')[:200] + '...' if post_data.get('selftext', '') else '',
                                'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
                                'source': 'Reddit r/CryptoCurrency',
                                'published_at': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                                'sentiment': post_data.get('score', 0) / 100  # Normalize score
                            })
                    
                    news_data = {
                        'articles': articles,
                        'source': 'reddit',
                        'count': len(articles)
                    }
                    
                    self._save_to_cache(cache_key, news_data)
                    return news_data
        except Exception as e:
            print(f"Reddit API error: {e}")
        
        # Return empty results if all fails
        return {'articles': [], 'source': 'none', 'count': 0}
    
    def get_market_stats(self, top_n=10):
        """Get market stats for top cryptocurrencies"""
        cache_key = f"market_stats_{top_n}"
        cached_data = self._try_from_cache(cache_key, max_age_hours=1)
        if cached_data:
            return cached_data
        
        # Try CoinGecko
        try:
            self._respect_rate_limit('coingecko')
            
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': top_n,
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '24h'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                market_data = {
                    'coins': data,
                    'source': 'coingecko',
                    'timestamp': datetime.now().isoformat()
                }
                
                self._save_to_cache(cache_key, market_data)
                return market_data
        except Exception as e:
            print(f"CoinGecko market stats error: {e}")
        
        # Try CoinCap as fallback
        try:
            self._respect_rate_limit('coincap')
            
            url = "https://api.coincap.io/v2/assets"
            params = {
                'limit': top_n
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                coins = data.get('data', [])
                
                # Format to match CoinGecko structure
                formatted_coins = []
                for coin in coins:
                    formatted_coins.append({
                        'id': coin.get('id', ''),
                        'symbol': coin.get('symbol', '').lower(),
                        'name': coin.get('name', ''),
                        'current_price': float(coin.get('priceUsd', 0)),
                        'market_cap': float(coin.get('marketCapUsd', 0)),
                        'market_cap_rank': int(coin.get('rank', 0)),
                        'price_change_percentage_24h': float(coin.get('changePercent24Hr', 0)),
                        'total_volume': float(coin.get('volumeUsd24Hr', 0))
                    })
                
                market_data = {
                    'coins': formatted_coins,
                    'source': 'coincap',
                    'timestamp': datetime.now().isoformat()
                }
                
                self._save_to_cache(cache_key, market_data)
                return market_data
        except Exception as e:
            print(f"CoinCap market stats error: {e}")
        
        # Return empty result if all fails
        return {'coins': [], 'source': 'none', 'timestamp': datetime.now().isoformat()}
