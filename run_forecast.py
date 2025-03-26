import argparse
import json
import os
from datetime import datetime
import time
import sys

# Make sure our modules are in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_model import CryptoMLModel
from discord_bot import DiscordLogger

def save_forecast(forecast, output_dir='data/forecasts'):
    """Save forecast to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/forecast_{forecast['symbol'].replace('/', '_')}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(forecast, f, indent=2)
    
    print(f"Forecast saved to {filename}")
    return filename

def run_forecast(symbol='BTC/USD', days=3, send_to_discord=False, discord_webhook=None):
    """Run a forecast for the specified symbol and days"""
    print(f"Generating {days}-day forecast for {symbol}...")
    
    # Initialize model
    model = CryptoMLModel()
    
    # Add Discord logger if requested
    discord_logger = None
    if send_to_discord and discord_webhook:
        # For this script we'll use a simple webhook-based logger
        # that doesn't require the full bot setup
        discord_logger = SimpleWebhookLogger(discord_webhook)
        model.discord_logger = discord_logger
    
    # Load model
    if not model.load_model():
        print("Warning: No saved model found. Results may not be accurate.")
    
    # Generate forecast
    start_time = time.time()
    forecast = model.get_forecast(symbol, days)
    
    if 'error' in forecast:
        print(f"Error generating forecast: {forecast['error']}")
        return None
    
    # Print forecast details
    print(f"\nForecast for {symbol} (next {days} days):")
    print(f"Current price: ${forecast['current_price']:.2f}")
    print(f"Generated at: {forecast['generated_at']}")
    print(f"Model version: {forecast['model_version']}")
    print(f"Trust score: {forecast['trust_score']:.2%}")
    
    print("\nDay-by-day forecast:")
    for day in forecast['forecasts']:
        print(f"Day {day['day']} ({day['date']}): {day['prediction'].upper()} " +
              f"({day['confidence']*100:.1f}% confidence)")
    
    # Save forecast
    save_forecast(forecast)
    
    # Send to Discord if requested
    if send_to_discord and discord_logger:
        days_forecast = "\n".join([
            f"Day {day['day']} ({day['date']}): {day['prediction'].upper()} " +
            f"({day['confidence']*100:.1f}% confidence)"
            for day in forecast['forecasts']
        ])
        
        discord_logger.send_log(
            f"🔮 **{days}-Day Forecast for {symbol}**\n" +
            f"Current price: ${forecast['current_price']:.2f}\n" +
            f"Trust score: {forecast['trust_score']*100:.1f}%\n\n" +
            f"{days_forecast}\n\n" +
            f"_This forecast is for educational purposes only. Not financial advice._",
            level="INFO"
        )
    
    print(f"\nForecast completed in {time.time() - start_time:.2f} seconds")
    return forecast

class SimpleWebhookLogger:
    """Simple Discord logger that uses webhooks instead of the bot"""
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send_log(self, message, level="INFO", chart=None):
        """Send a message to Discord webhook"""
        import requests
        
        # Format based on level
        if level == "ERROR":
            emoji = "❌ "
            color = 0xFF0000  # Red
        elif level == "WARNING":
            emoji = "⚠️ "
            color = 0xFFA500  # Orange
        elif level == "SUCCESS":
            emoji = "✅ "
            color = 0x00FF00  # Green
        else:
            emoji = "ℹ️ "
            color = 0x0000FF  # Blue
        
        # Create payload
        payload = {
            "content": None,
            "embeds": [
                {
                    "description": f"{emoji}{message}",
                    "color": color,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        # Send to webhook
        try:
            response = requests.post(
                self.webhook_url,
                json=payload
            )
            response.raise_for_status()
            print(f"Discord message sent ({level})")
            return True
        except Exception as e:
            print(f"Error sending Discord message: {e}")
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate crypto forecast')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Cryptocurrency symbol')
    parser.add_argument('--days', type=int, default=3, help='Number of days to forecast')
    parser.add_argument('--discord', action='store_true', help='Send forecast to Discord')
    parser.add_argument('--webhook', type=str, help='Discord webhook URL for sending forecast')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Run forecast
    run_forecast(
        symbol=args.symbol,
        days=args.days,
        send_to_discord=args.discord,
        discord_webhook=args.webhook
    )
