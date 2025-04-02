import os
import json
import time
import threading
import asyncio
import argparse
from datetime import datetime, timedelta
import pandas as pd
import tracemalloc  # Add tracemalloc module
from indicator_script import CryptoIndicators
from ml_model import CryptoMLModel
from news_sentiment import NewsSentimentAnalyzer
from discord_bot import CryptoDiscordBot, DiscordLogger

# Enable tracemalloc tracking
tracemalloc.start(25)  # Number of frames to capture in tracebacks

class CryptoPredictionSystem:
    def __init__(self, config=None):
        # Load configuration
        self.config = config or self.load_config()
        
        # API keys are now optional since we use free APIs
        self.news_api_key = None  # No longer required
        self.crypto_api_key = None  # No longer required
        self.discord_token = ''
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/feedback', exist_ok=True)
        
        # Create logger first so other components can use it
        self.discord_logger = None
        if self.discord_token:
            try:
                # Create a temporary logger that will be replaced with actual one when bot starts
                self.discord_logger = TempDiscordLogger()
            except Exception as e:
                print(f"Warning: Couldn't initialize logger: {e}")
        
        # Initialize components with error handling
        try:
            # No API key required now
            self.indicators = CryptoIndicators(discord_logger=self.discord_logger)
            print("Technical indicators component initialized successfully")
        except Exception as e:
            print(f"Error initializing indicators component: {e}")
            self.indicators = None
            
        try:
            self.ml_model = CryptoMLModel(discord_logger=self.discord_logger)
            # Load or train ML model
            if not self.ml_model.load_model():
                print("Training new ML model...")
                self.ml_model.train_model()
            print("ML model component initialized successfully")
        except Exception as e:
            print(f"Error initializing ML model component: {e}")
            self.ml_model = None
            
        try:
            # No API key required now
            self.news_analyzer = NewsSentimentAnalyzer()
            print("News analyzer component initialized successfully")
        except Exception as e:
            print(f"Error initializing news analyzer component: {e}")
            self.news_analyzer = None
        
        # Discord bot (initialized but not started yet)
        try:
            self.discord_bot = None if not self.discord_token else CryptoDiscordBot(self, self.discord_token)
            if self.discord_bot:
                print("Discord bot component initialized successfully")
                # Replace the temporary logger with the actual Discord bot logger
                self.discord_logger = self.discord_bot.logger
                
                # Update logger references in components
                if self.indicators:
                    self.indicators.discord_logger = self.discord_logger
                if self.ml_model:
                    self.ml_model.discord_logger = self.discord_logger
                
                # Start the Discord bot asynchronously
                self.discord_bot.start_async()  # Pe8a4
        except Exception as e:
            print(f"Error initializing Discord bot component: {e}")
            self.discord_bot = None
        
        # Set up a weekly learning check
        self.last_learning_check = datetime.now()
        
        # System status reporting
        print(f"System initialized with:")
        print(f"  - Indicators: {'✅' if self.indicators else '❌'}")
        print(f"  - ML Model: {'✅' if self.ml_model else '❌'}")
        print(f"  - News Analyzer: {'✅' if self.news_analyzer else '❌'}")
        print(f"  - Discord Bot: {'✅' if self.discord_bot else '❌'}")
    
    def load_config(self, config_path='config.json'):
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        return {}
    
    def save_config(self, config_path='config.json'):
        """Save configuration to JSON file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def _check_learning_progress(self):
        """Check if we need to evaluate learning progress"""
        # Check if it's been a week since our last check
        days_since_check = (datetime.now() - self.last_learning_check).days
        
        if days_since_check >= 7 and self.ml_model:
            print("Performing weekly learning evaluation...")
            
            # Evaluate if retraining is needed based on feedback
            if hasattr(self.ml_model, 'feedback_samples') and len(self.ml_model.feedback_samples) > 0:
                self.ml_model._evaluate_retraining_need()
            
            # Check learning progress
            if self.discord_logger:
                learning_history = self.ml_model.learning_history
                iterations = learning_history.get('learning_iterations', 0)
                
                self.discord_logger.send_log_sync(
                    f"Weekly Learning Status: {iterations} total learning iterations", 
                    level="LEARNING"
                )
                
                # Generate performance chart
                if hasattr(self.ml_model, '_generate_performance_chart'):
                    self.ml_model._generate_performance_chart()
            
            # Update last check time
            self.last_learning_check = datetime.now()
    
    def run_prediction_cycle(self, symbol='BTC/USD', timeframe='1h'):
        """Run a complete prediction cycle with graceful failure handling"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n=== Starting prediction cycle at {timestamp} ===")
        
        # Check if we need to do a learning evaluation
        self._check_learning_progress()
        
        # Step 1: Get technical indicators and signals
        indicator_signals = None
        if self.indicators:
            print("Analyzing technical indicators...")
            try:
                indicator_signals = self.indicators.run_analysis(symbol, timeframe)
                if not indicator_signals:
                    print("Failed to get indicator signals.")
            except Exception as e:
                print(f"Error in technical analysis: {e}")
        else:
            print("Technical indicators component not available.")
        
        # If we don't have indicator signals, we can't proceed
        if not indicator_signals:
            print("Critical component (indicators) failed. Cannot generate prediction.")
            return None
        
        # Step 2: Get news sentiment (optional component)
        news_sentiment = {"overall_sentiment": "neutral", "sentiment_score": 0.0, "data_available": False}
        if self.news_analyzer:
            print("Analyzing news sentiment...")
            try:
                news_sentiment = self.news_analyzer.get_news_sentiment("bitcoin", days=1, max_articles=3)
                if not news_sentiment.get('data_available', False):
                    print("No news data available, proceeding with neutral sentiment.")
            except Exception as e:
                print(f"Error in news sentiment analysis: {e}")
                print("Continuing with neutral sentiment.")
        else:
            print("News analyzer component not available. Continuing with neutral sentiment.")
        
        # Step 3: ML model prediction (optional but valuable component)
        ml_prediction = {"prediction": "neutral", "confidence": 0.5, "buy_probability": 0.5, "sell_probability": 0.5}
        if self.ml_model:
            print("Running ML prediction...")
            try:
                ml_prediction = self.ml_model.predict(indicator_signals)
                # Inject sentiment data if available
                if self.news_analyzer and news_sentiment.get('data_available'):
                    # Create a copy of the signals with sentiment data
                    signals_with_sentiment = indicator_signals.copy()
                    if 'components' not in signals_with_sentiment:
                        signals_with_sentiment['components'] = {}
                    signals_with_sentiment['components']['sentiment'] = {
                        'score': news_sentiment.get('sentiment_score', 0)
                    }
                    # Get new prediction with sentiment data
                    ml_prediction = self.ml_model.predict(signals_with_sentiment)
            except Exception as e:
                print(f"Error in ML prediction: {e}")
                print("Continuing with basic ML prediction.")
        else:
            print("ML model component not available. Using basic prediction rules.")
            # Simple rule-based fallback if ML model is not available
            rsi = indicator_signals.get('indicators', {}).get('rsi', 50)
            if rsi < 30:
                ml_prediction = {"prediction": "buy", "confidence": 0.6}
            elif rsi > 70:
                ml_prediction = {"prediction": "sell", "confidence": 0.6}
        
        # Step 4: Combine signals with sentiment for final prediction
        final_prediction = self.combine_predictions(indicator_signals, ml_prediction, news_sentiment)
        
        # Step 5: Save prediction to log
        self.log_prediction(final_prediction)
        
        # Log to Discord if available
        if self.discord_logger:
            log_message = f"New prediction: {final_prediction['signal'].upper()} with {final_prediction['confidence']*100:.1f}% confidence"
            if final_prediction.get('conflicting_signals', False):
                log_message += "\n⚠️ Warning: Conflicting signals detected!"
            
            self.discord_logger.send_log_sync(log_message, level="INFO")
        
        print(f"Final prediction: {final_prediction['signal']} (Confidence: {final_prediction['confidence']:.2f})")
        print(f"=== Prediction cycle completed ===\n")
        
        return final_prediction
    
    def combine_predictions(self, indicator_signals, ml_prediction, news_sentiment):
        """Combine various signals into a final prediction, adaptive to available data"""
        # Get direction and confidence from each source
        indicator_signal = indicator_signals.get('signal', 'neutral')
        indicator_confidence = indicator_signals.get('confidence', 0.5)
        
        ml_signal = ml_prediction.get('prediction', 'neutral')
        ml_confidence = ml_prediction.get('confidence', 0.5)
        
        # Convert string sentiment to numeric
        sentiment_score = news_sentiment.get('sentiment_score', 0)
        sentiment_available = news_sentiment.get('data_available', False)
        
        # Determine weights based on what's available
        if self.ml_model and self.news_analyzer and sentiment_available:
            # All components available
            indicator_weight = 0.4
            ml_weight = 0.4
            sentiment_weight = 0.2
        elif self.ml_model:
            # ML available but no sentiment
            indicator_weight = 0.5
            ml_weight = 0.5
            sentiment_weight = 0
        else:
            # Only indicators available
            indicator_weight = 1.0
            ml_weight = 0
            sentiment_weight = 0
        
        # Convert signals to numeric (-1 to 1)
        indicator_value = 1 if indicator_signal == 'buy' else (-1 if indicator_signal == 'sell' else 0)
        ml_value = 1 if ml_signal == 'buy' else (-1 if ml_signal == 'sell' else 0)
        
        # Weighted combination
        combined_value = (
            indicator_value * indicator_confidence * indicator_weight +
            ml_value * ml_confidence * ml_weight +
            sentiment_score * sentiment_weight
        )
        
        # Determine final signal
        if combined_value > 0.1:
            final_signal = 'buy'
        elif combined_value < -0.1:
            final_signal = 'sell'
        else:
            final_signal = 'neutral'
        
        # Calculate confidence (absolute value of combined, normalized to 0-1)
        confidence = min(abs(combined_value) * 2, 1.0)
        
        # Check for conflicting signals
        conflicting = False
        if ml_weight > 0 and indicator_weight > 0:
            conflicting = (indicator_value * ml_value < 0)
        if sentiment_weight > 0:
            if indicator_weight > 0:
                conflicting = conflicting or (indicator_value * sentiment_score < 0)
            if ml_weight > 0:
                conflicting = conflicting or (ml_value * sentiment_score < 0)
        
        components = {
            'indicator': {
                'signal': indicator_signal,
                'confidence': indicator_confidence
            }
        }
        
        if ml_weight > 0:
            components['ml_model'] = {
                'signal': ml_signal,
                'confidence': ml_confidence,
                'buy_probability': ml_prediction.get('buy_probability', 0),
                'sell_probability': ml_prediction.get('sell_probability', 0)
            }
            
        if sentiment_weight > 0:
            components['sentiment'] = {
                'score': sentiment_score,
                'summary': news_sentiment.get('summary', '')
            }
        
        final_prediction = {
            'timestamp': datetime.now().isoformat(),
            'signal': final_signal,
            'confidence': confidence,
            'price': indicator_signals.get('price', 0),
            'conflicting_signals': conflicting,
            'components': components,
            'used_ml': ml_weight > 0,
            'used_sentiment': sentiment_weight > 0
        }
        
        return final_prediction
    
    def log_prediction(self, prediction):
        """Save prediction to log file"""
        log_file = f"logs/predictions_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Read existing log if it exists
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []
        
        # Add new prediction
        logs.append(prediction)
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def evaluate_past_predictions(self, days=7):
        """Evaluate the accuracy of past predictions and use for learning"""
        print(f"Starting automated evaluation of the past {days} days...")
        
        try:
            # Use the AutoEvaluator to evaluate past predictions
            from auto_evaluator import AutoEvaluator
            evaluator = AutoEvaluator(self)
            
            # Run the evaluation and add feedback to the system
            results = evaluator.run_evaluation(days=days, symbol='BTC/USD', add_feedback=True)
            
            if results is not None:
                # Log the evaluation results
                print(f"Evaluation complete. Accuracy statistics:")
                print(f"1-hour: {results['correct_1h'].mean():.2%}")
                print(f"4-hour: {results['correct_4h'].mean():.2%}")
                print(f"24-hour: {results['correct_24h'].mean():.2%}")
                
                # Log to Discord if available
                if self.discord_logger:
                    self.discord_logger.send_log_sync(
                        f"Completed auto-evaluation of {len(results)} predictions over {days} days.\n" +
                        f"Accuracy: 1h={results['correct_1h'].mean():.1%}, 4h={results['correct_4h'].mean():.1%}, 24h={results['correct_24h'].mean():.1%}",
                        level="LEARNING"
                    )
                
                # If we have ML model, check if we need to trigger learning
                if self.ml_model and hasattr(self.ml_model, 'feedback_samples'):
                    if len(self.ml_model.feedback_samples) >= self.ml_model.min_feedback_samples:
                        print("Enough feedback samples available, evaluating learning needs...")
                        self.ml_model._evaluate_retraining_need()
                    else:
                        print(f"Only {len(self.ml_model.feedback_samples)} feedback samples available. Need {self.ml_model.min_feedback_samples} for retraining.")
                        
                return results
            else:
                print("No evaluation results available.")
                return None
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_actual_with_predicted(self, prediction, actual_outcome):
        """Compare prediction with actual outcome and use for learning"""
        if not prediction:
            print("No prediction provided for comparison")
            return False
        
        print(f"Comparing prediction ({prediction['signal']}) with actual outcome ({actual_outcome})")
        
        # Convert outcome to appropriate format
        if isinstance(actual_outcome, bool):
            # True = price went up (buy), False = price went down (sell)
            outcome_str = "buy" if actual_outcome else "sell"
        elif isinstance(actual_outcome, str):
            outcome_str = actual_outcome.lower()
            if outcome_str not in ['buy', 'sell', 'neutral']:
                print(f"Invalid outcome string: {outcome_str}")
                return False
        else:
            print(f"Invalid outcome type: {type(actual_outcome)}")
            return False
        
        # Check if prediction was correct
        correct = prediction['signal'] == outcome_str
        print(f"Prediction was {'correct' if correct else 'incorrect'}")
        
        # Update indicator performance if available
        if self.indicators:
            try:
                self.indicators.update_indicator_performance(outcome_str)
                print("Updated indicator performance")
            except Exception as e:
                print(f"Error updating indicator performance: {e}")
        
        # Add feedback to ML model if available
        if self.ml_model:
            try:
                self.ml_model.add_feedback(prediction, outcome_str)
                print("Added feedback to ML model")
            except Exception as e:
                print(f"Error adding feedback to ML model: {e}")
        
        return correct
    
    def run_backtest(self, start_date=None, end_date=None, symbol='BTC/USD', timeframe='1h'):
        """Run a backtest of the prediction system on historical data"""
        print("Backtest functionality to be implemented")
        # TODO: Implement backtest functionality with historical data
    
    def run(self, interval_minutes=60, cycles=None, discord=False):
        """Run the prediction system continuously with error handling"""
        print(f"Starting crypto prediction system with {interval_minutes} minute interval")
        
        # Start Discord bot in a separate thread if requested
        discord_thread = None
        if discord and self.discord_bot:
            print("Starting Discord bot...")
            discord_thread = threading.Thread(target=self.discord_bot.run, daemon=True)
            discord_thread.start()
        
        cycle_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                try:
                    # Run prediction cycle with error handling
                    result = self.run_prediction_cycle()
                    if result:
                        consecutive_errors = 0  # Reset error counter on success
                    else:
                        consecutive_errors += 1
                        print(f"Warning: Prediction cycle failed. Consecutive errors: {consecutive_errors}")
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error in prediction cycle: {e}")
                    print(f"Consecutive errors: {consecutive_errors}")
                
                # Check if we should abort due to too many errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}). Stopping prediction system.")
                    break
                
                # Check if we've reached the specified number of cycles
                cycle_count += 1
                if cycles and cycle_count >= cycles:
                    print(f"Completed {cycles} prediction cycles.")
                    break
                
                # Wait for the next interval
                print(f"Waiting {interval_minutes} minutes until next prediction...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("Prediction system stopped by user.")
        finally:
            print("Shutting down prediction system...")
            # Save any pending data
            if self.ml_model:
                self.ml_model._save_learning_history()
            
            # If Discord bot is running, give it time to shut down gracefully
            if discord_thread and discord_thread.is_alive():
                print("Waiting for Discord bot to shut down...")
                time.sleep(5)
    
    def print_memory_stats(self):
        """Print memory statistics and top allocations"""
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")
        
        print("\nTop 10 memory allocations:")
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        for i, stat in enumerate(top_stats[:10], 1):
            print(f"#{i}: {stat}")


# Temporary logger for startup before Discord bot is running
class TempDiscordLogger:
    def __init__(self):
        self.pending_messages = []
    
    def send_log(self, message, level="INFO", chart=None):
        """Store logs until real logger is available"""
        print(f"[{level}] {message} (Discord logging queued)")
        self.pending_messages.append((message, level, chart))
    
    def send_log_sync(self, message, level="INFO", chart=None):
        """Store logs until real logger is available"""
        print(f"[{level}] {message} (Discord logging queued)")
        self.pending_messages.append((message, level, chart))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Crypto Prediction System')
    parser.add_argument('--discord', action='store_true', help='Start with Discord bot')
    parser.add_argument('--interval', type=int, default=60, help='Prediction interval in minutes')
    parser.add_argument('--cycles', type=int, help='Number of prediction cycles to run')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Cryptocurrency symbol to analyze')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate past predictions')
    parser.add_argument('--days', type=int, default=7, help='Number of days to include in evaluation')
    parser.add_argument('--learn', action='store_true', help='Force learning evaluation and retraining')
    parser.add_argument('--memory', action='store_true', help='Print memory statistics')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    system = CryptoPredictionSystem()
    
    if args.evaluate:
        system.evaluate_past_predictions(days=args.days)
    elif args.learn and system.ml_model:
        # Force a learning evaluation
        print("Forcing learning evaluation...")
        system.ml_model._evaluate_retraining_need()
    elif args.memory:
        # Print memory stats when requested
        system.print_memory_stats()
    else:
        # Run a single prediction cycle or continuous mode
        if args.cycles == 1:
            system.run_prediction_cycle(symbol=args.symbol)
            # Print memory stats after run for debugging
            if tracemalloc.is_tracing():
                system.print_memory_stats()
        else:
            system.run(interval_minutes=args.interval, cycles=args.cycles, discord=args.discord)
