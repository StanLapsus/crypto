import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import argparse
import time
import traceback

class AutoEvaluator:
    """Automatically evaluate past predictions by comparing with actual price data"""
    
    def __init__(self, system=None):
        self.system = system
        self.data_dir = 'data'
        self.logs_dir = 'logs'
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def load_predictions(self, days=7):
        """Load prediction logs from the specified period"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_predictions = []
        
        # Load prediction logs from each day
        for i in range(days + 1):
            date = end_date - timedelta(days=i)
            log_file = f"{self.logs_dir}/predictions_{date.strftime('%Y%m%d')}.json"
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        predictions = json.load(f)
                        all_predictions.extend(predictions)
                    except json.JSONDecodeError:
                        continue
        
        # Convert to DataFrame
        if all_predictions:
            df = pd.DataFrame(all_predictions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            return df
        else:
            return None
    
    def fetch_historical_prices(self, symbol='BTC/USD', days=7):
        """Fetch historical prices for the specified period"""
        try:
            # Clean symbol for API
            clean_symbol = symbol.replace('/', '').lower()
            
            # Try CoinGecko API
            url = f"https://api.coingecko.com/api/v3/coins/{clean_symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
                
            data = response.json()
            
            # Extract price data
            prices = data.get('prices', [])
            if not prices:
                raise Exception("No price data returned")
                
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical prices: {e}")
            return None
    
    def evaluate_prediction(self, prediction, price_df):
        """Evaluate if a prediction was correct based on subsequent price movement"""
        timestamp = pd.to_datetime(prediction['timestamp'])
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        # Find next price data points after the prediction
        future_prices = price_df[price_df['timestamp'] > timestamp].sort_values('timestamp')
        
        if len(future_prices) < 2:
            print(f"Not enough future price data to evaluate prediction at {timestamp}")
            return None
        
        # Get prices at different timeframes for evaluation
        next_1h = future_prices.iloc[min(1, len(future_prices)-1)]
        next_4h = future_prices.iloc[min(4, len(future_prices)-1)]
        next_24h = future_prices.iloc[min(24, len(future_prices)-1)]
        
        # Calculate price changes
        price_change_1h = (next_1h['price'].iloc[0] - future_prices.iloc[0]['price'].iloc[0]) / future_prices.iloc[0]['price'].iloc[0]
        price_change_4h = (next_4h['price'].iloc[0] - future_prices.iloc[0]['price'].iloc[0]) / future_prices.iloc[0]['price'].iloc[0]
        price_change_24h = (next_24h['price'].iloc[0] - future_prices.iloc[0]['price'].iloc[0]) / future_prices.iloc[0]['price'].iloc[0]
        
        # Determine if prediction was correct
        actual_1h = 'buy' if price_change_1h > 0 else ('sell' if price_change_1h < 0 else 'neutral')
        actual_4h = 'buy' if price_change_4h > 0 else ('sell' if price_change_4h < 0 else 'neutral')
        actual_24h = 'buy' if price_change_24h > 0 else ('sell' if price_change_24h < 0 else 'neutral')
        
        correct_1h = signal == actual_1h
        correct_4h = signal == actual_4h
        correct_24h = signal == actual_24h
        
        # Create evaluation result
        result = {
            'prediction_timestamp': timestamp.isoformat(),
            'prediction_signal': signal,
            'prediction_confidence': confidence,
            'price_change_1h': price_change_1h,
            'price_change_4h': price_change_4h,
            'price_change_24h': price_change_24h,
            'actual_1h': actual_1h,
            'actual_4h': actual_4h,
            'actual_24h': actual_24h,
            'correct_1h': correct_1h,
            'correct_4h': correct_4h,
            'correct_24h': correct_24h,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def run_evaluation(self, days=7, symbol='BTC/USD', add_feedback=True):
        """Run evaluation on past predictions and optionally add feedback to the system"""
        print(f"Running auto-evaluation for the past {days} days...")
        
        # Load predictions
        prediction_df = self.load_predictions(days)
        if prediction_df is None or len(prediction_df) == 0:
            print("No predictions found for the specified period.")
            return None
        
        # Fetch historical prices
        price_df = self.fetch_historical_prices(symbol, days)
        if price_df is None or len(price_df) == 0:
            print("No price data available for evaluation.")
            return None
        
        # Ensure we're using the correct method name when evaluating indicators
        if hasattr(self.system, 'indicators') and self.system.indicators:
            # Fix any potential typo from run_analyeis to run_analysis
            if hasattr(self.system.indicators, 'run_analysis'):
                # Use the correct method name
                indicator_data = self.system.indicators.run_analysis(symbol, '1h')
        
        # Evaluate each prediction
        evaluations = []
        for _, prediction in prediction_df.iterrows():
            result = self.evaluate_prediction(prediction, price_df)
            if result:
                evaluations.append(result)
                
                # Add feedback to system if requested
                if add_feedback and self.system:
                    print(f"Adding feedback for prediction at {result['prediction_timestamp']}")
                    
                    # Use 4h outcome as feedback since it's a good medium-term horizon
                    self.system.compare_actual_with_predicted(prediction, result['actual_4h'])
                    
                    # Don't add feedback too quickly to avoid overloading APIs
                    time.sleep(1)
        
        # Convert evaluations to DataFrame
        if evaluations:
            eval_df = pd.DataFrame(evaluations)
            
            # Calculate accuracy at different timeframes
            accuracy_1h = eval_df['correct_1h'].mean()
            accuracy_4h = eval_df['correct_4h'].mean()
            accuracy_24h = eval_df['correct_24h'].mean()
            
            print(f"Evaluation complete. Results:")
            print(f"Total predictions evaluated: {len(eval_df)}")
            print(f"1-hour accuracy: {accuracy_1h:.2%}")
            print(f"4-hour accuracy: {accuracy_4h:.2%}")
            print(f"24-hour accuracy: {accuracy_24h:.2%}")
            
            # Save results
            results_file = f"{self.data_dir}/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'days_evaluated': days,
                        'symbol': symbol,
                        'total_predictions': len(eval_df),
                        'accuracy_1h': float(accuracy_1h),
                        'accuracy_4h': float(accuracy_4h),
                        'accuracy_24h': float(accuracy_24h)
                    },
                    'evaluations': evaluations
                }, f, indent=2)
            
            print(f"Evaluation results saved to {results_file}")
            
            return eval_df
        else:
            print("No evaluations could be completed.")
            return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Auto-evaluate crypto predictions')
    parser.add_argument('--days', type=int, default=7, help='Number of days to evaluate')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Cryptocurrency symbol')
    parser.add_argument('--no-feedback', action='store_true', help='Don\'t add feedback to system')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize evaluator without system since we're running standalone
    evaluator = AutoEvaluator()
    
    # Run evaluation
    evaluator.run_evaluation(
        days=args.days,
        symbol=args.symbol,
        add_feedback=not args.no_feedback
    )
