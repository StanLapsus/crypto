import discord
from discord.ext import commands, tasks
import os
import asyncio
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import pandas as pd
import traceback

class DiscordLogger:
    """Helper class for logging to Discord channels"""
    def __init__(self, bot, log_channel_id=None):
        self.bot = bot
        self.log_channel_id = log_channel_id
        self.message_queue = []
        self.is_sending = False
        self.tasks = []  # Track created tasks
        
    async def send_log(self, message, level="INFO", chart=None):
        """Send log message to Discord channel"""
        if not self.log_channel_id:
            return
            
        try:
            channel = self.bot.get_channel(self.log_channel_id)
            if not channel:
                return
                
            # Format message with appropriate emoji based on level
            emoji = "ℹ️"
            color = discord.Color.blue()
            
            if level == "ERROR":
                emoji = "❌"
                color = discord.Color.red()
            elif level == "WARNING":
                emoji = "⚠️"
                color = discord.Color.orange()
            elif level == "SUCCESS":
                emoji = "✅"
                color = discord.Color.green()
            elif level == "LEARNING":
                emoji = "🧠"
                color = discord.Color.purple()
            
            # Create an embed for better formatting
            embed = discord.Embed(
                description=f"{emoji} {message}",
                color=color,
                timestamp=datetime.now()
            )
            
            if chart:
                # If a chart is provided, send it as a file attachment
                await channel.send(embed=embed, file=discord.File(chart, 'chart.png'))
            else:
                # Otherwise just send the embed
                await channel.send(embed=embed)
                
        except Exception as e:
            print(f"Error sending log to Discord: {e}")
            
    def send_log_sync(self, message, level="INFO", chart=None):
        """Add message to queue for async sending"""
        self.message_queue.append((message, level, chart))
        if not self.is_sending:
            # Create and store the task reference to prevent warnings
            task = asyncio.create_task(self._process_queue())
            self.tasks.append(task)
            # Add done callback to remove the task when completed
            task.add_done_callback(lambda t: self.tasks.remove(t) if t in self.tasks else None)
            
    async def _process_queue(self):
        """Process queued messages"""
        if self.is_sending:
            return
            
        self.is_sending = True
        try:
            while self.message_queue:
                message, level, chart = self.message_queue.pop(0)
                await self.send_log(message, level, chart)
                await asyncio.sleep(0.5)  # Prevent rate limiting
        finally:
            self.is_sending = False

    def log(self, message, level="INFO", chart=None):
        """Non-async version that queues the message"""
        self.send_log_sync(message, level, chart)

class CryptoDiscordBot:
    def __init__(self, prediction_system, token=None):
        self.token = token or os.environ.get('DISCORD_BOT_TOKEN')
        if not self.token:
            raise ValueError("Discord bot token is required. Set DISCORD_BOT_TOKEN env variable or pass token in constructor.")
        
        self.prediction_system = prediction_system
        
        # Create Discord bot instance
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='!crypto ', intents=intents)
        
        # Logging channels (hardcoded IDs)
        self.alert_channel_id = 1352649930501525587 # Replace with actual channel ID
        self.log_channel_id = 1352649930501525587   # Replace with actual channel ID
        self.learning_channel_id = 1352649930501525587  # Replace with actual channel ID
        
        # Create logger instance
        self.logger = DiscordLogger(self.bot, self.log_channel_id)
        
        # Make logger available to the prediction system components
        if self.prediction_system:
            if hasattr(self.prediction_system, 'indicators') and self.prediction_system.indicators:
                self.prediction_system.indicators.discord_logger = self.logger
            if hasattr(self.prediction_system, 'ml_model') and self.prediction_system.ml_model:
                self.prediction_system.ml_model.discord_logger = self.logger
        
        # Register commands
        self.register_commands()
        
        # Alert channels configuration (channel_id: min_confidence)
        self.alert_channels = {}
        self.load_alert_channels()
        
        # Start background tasks when bot is ready
        @self.bot.event
        async def on_ready():
            print(f"Discord bot logged in as {self.bot.user}")
            # Send startup message
            await self.logger.send_log(f"🚀 Crypto Trading Bot is now online and ready!", "SUCCESS")
            self.check_for_predictions.start()
            self.check_for_learning_updates.start()
    
    def register_commands(self):
        """Register bot commands"""
        
        @self.bot.command(name="predict", help="Run a prediction for a specific crypto")
        async def predict(ctx, symbol="BTC/USD"):
            """Run a prediction and return results"""
            await ctx.send(f"Running prediction for {symbol}...")
            
            try:
                # Run prediction cycle
                prediction = self.prediction_system.run_prediction_cycle(symbol=symbol)
                
                if not prediction:
                    await ctx.send("Failed to generate prediction. Check logs for details.")
                    return
                
                # Create an embed message with the prediction
                embed = discord.Embed(
                    title=f"Crypto Prediction for {symbol}",
                    color=self.get_color_for_signal(prediction.get('signal', 'neutral')),
                    timestamp=datetime.now()
                )
                
                embed.add_field(
                    name="Signal", 
                    value=prediction.get('signal', 'neutral').upper(), 
                    inline=True
                )
                embed.add_field(
                    name="Confidence", 
                    value=f"{prediction.get('confidence', 0)*100:.1f}%", 
                    inline=True
                )
                embed.add_field(
                    name="Price", 
                    value=f"${prediction.get('price', 0):,.2f}", 
                    inline=True
                )
                
                components = prediction.get('components', {})
                
                # Technical indicators
                indicator_data = components.get('indicator', {})
                embed.add_field(
                    name="Technical Indicators", 
                    value=f"Signal: {indicator_data.get('signal', 'neutral')}\nConfidence: {indicator_data.get('confidence', 0)*100:.1f}%", 
                    inline=True
                )
                
                # ML model
                ml_data = components.get('ml_model', {})
                embed.add_field(
                    name="ML Model", 
                    value=f"Signal: {ml_data.get('signal', 'neutral')}\nConfidence: {ml_data.get('confidence', 0)*100:.1f}%", 
                    inline=True
                )
                
                # Sentiment
                sentiment_data = components.get('sentiment', {})
                embed.add_field(
                    name="News Sentiment", 
                    value=f"Score: {sentiment_data.get('score', 0):.2f}\n{sentiment_data.get('summary', '')}",
                    inline=False
                )
                
                # Add warning for conflicting signals
                if prediction.get('conflicting_signals', False):
                    embed.add_field(
                        name="⚠️ Warning", 
                        value="Conflicting signals detected! Use caution.", 
                        inline=False
                    )
                
                embed.set_footer(text="Crypto Prediction Bot | Use !crypto help for more commands")
                
                await ctx.send(embed=embed)
                
            except Exception as e:
                await ctx.send(f"Error generating prediction: {str(e)}")
        
        @self.bot.command(name="forecast", help="Generate a forecast for the next few days")
        async def forecast(ctx, symbol="BTC/USD", days: int = 3):
            """Generate and show a forecast for the specified cryptocurrency"""
            if days < 1 or days > 7:
                await ctx.send("Please specify between 1 and 7 days for the forecast.")
                return
                
            await ctx.send(f"Generating {days}-day forecast for {symbol}. This may take a moment...")
            
            try:
                # Check if ML model is available
                if not self.prediction_system.ml_model:
                    await ctx.send("ML model component not available. Cannot generate forecast.")
                    return
                    
                # Call the forecast method
                forecast_result = self.prediction_system.ml_model.get_forecast(symbol, days)
                
                if 'error' in forecast_result:
                    await ctx.send(f"Error generating forecast: {forecast_result['error']}")
                    return
                
                # Create a nice embed for the forecast
                embed = discord.Embed(
                    title=f"{days}-Day Forecast for {symbol}",
                    description=f"Current price: ${forecast_result.get('current_price', 0):,.2f}",
                    color=discord.Color.gold(),
                    timestamp=datetime.fromisoformat(forecast_result.get('generated_at', datetime.now().isoformat()))
                )
                
                # Add model info
                embed.add_field(
                    name="Model Info",
                    value=f"Version: {forecast_result.get('model_version', 'Unknown')}\nTrust Score: {forecast_result.get('trust_score', 0)*100:.1f}%",
                    inline=False
                )
                
                # Add each day's forecast
                for day in forecast_result.get('forecasts', []):
                    color = "🟢" if day['prediction'] == 'buy' else "🔴"
                    embed.add_field(
                        name=f"Day {day['day']} ({day['date']})",
                        value=f"{color} {day['prediction'].upper()} ({day['confidence']*100:.1f}% confidence)\nDirection: {day['price_direction']}",
                        inline=True
                    )
                
                # Add disclaimer
                embed.set_footer(text="⚠️ This forecast is for educational purposes only. Not financial advice.")
                
                # Generate a chart visualization
                chart_buffer = await self.generate_forecast_chart(forecast_result)
                
                if chart_buffer:
                    await ctx.send(embed=embed, file=discord.File(chart_buffer, 'forecast.png'))
                else:
                    await ctx.send(embed=embed)
                
                # Add tracking message for later evaluation
                await ctx.send("👁️ React with 👍 or 👎 to provide feedback on this forecast when the time comes!")
                
            except Exception as e:
                error_msg = f"Error generating forecast: {str(e)}"
                traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                print(error_msg)
                print(traceback_str)
                await ctx.send(error_msg)
        
        @self.bot.command(name="evaluate", help="Evaluate a previous prediction")
        async def evaluate(ctx, outcome: str = None):
            """Evaluate a previous prediction to help the model learn"""
            if not outcome or outcome.lower() not in ['buy', 'sell', 'neutral']:
                await ctx.send("Please specify the actual outcome as 'buy', 'sell', or 'neutral'.")
                return
                
            await ctx.send("Looking for recent predictions to evaluate...")
            
            try:
                # Get recent predictions for this channel
                today = datetime.now().strftime('%Y%m%d')
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                
                recent_predictions = []
                
                # Check today's and yesterday's logs
                for date in [today, yesterday]:
                    log_file = f"logs/predictions_{date}.json"
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                predictions = json.load(f)
                                # Add day marker to predictions
                                for p in predictions:
                                    p['log_date'] = date
                                recent_predictions.extend(predictions)
                        except:
                            pass
                
                if not recent_predictions:
                    await ctx.send("No recent predictions found to evaluate.")
                    return
                
                # Get the most recent prediction
                recent_predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                latest_prediction = recent_predictions[0]
                
                # Send info about the prediction being evaluated
                embed = discord.Embed(
                    title="Prediction Evaluation",
                    description=f"Evaluating prediction from {datetime.fromisoformat(latest_prediction.get('timestamp', '')).strftime('%Y-%m-%d %H:%M')}",
                    color=self.get_color_for_signal(latest_prediction.get('signal', 'neutral'))
                )
                
                embed.add_field(
                    name="Original Prediction",
                    value=f"{latest_prediction.get('signal', 'neutral').upper()} with {latest_prediction.get('confidence', 0)*100:.1f}% confidence",
                    inline=False
                )
                
                embed.add_field(
                    name="Actual Outcome",
                    value=f"{outcome.upper()}",
                    inline=False
                )
                
                # Determine if prediction was correct
                correct = latest_prediction.get('signal', 'neutral').lower() == outcome.lower()
                result = "✅ Correct" if correct else "❌ Incorrect"
                
                embed.add_field(
                    name="Result",
                    value=result,
                    inline=False
                )
                
                await ctx.send(embed=embed)
                
                # Add feedback to the model
                if self.prediction_system.ml_model:
                    self.prediction_system.ml_model.add_feedback(
                        latest_prediction, 
                        outcome, 
                        human_feedback={"user_id": ctx.author.id, "channel_id": ctx.channel.id}
                    )
                    
                    await ctx.send("Thank you for your feedback! The model will use this to improve.")
                    
                    # Also update indicator weights
                    if self.prediction_system.indicators:
                        self.prediction_system.indicators.update_indicator_performance(latest_prediction, outcome)
                
            except Exception as e:
                await ctx.send(f"Error evaluating prediction: {str(e)}")
        
        @self.bot.command(name="learning", help="Show model learning progress")
        async def learning(ctx):
            """Show the model's learning progress and current status"""
            try:
                # Check if ML model is available
                if not self.prediction_system.ml_model:
                    await ctx.send("ML model component not available.")
                    return
                    
                # Get learning history from the model
                learning_history = self.prediction_system.ml_model.learning_history
                
                # Create embed
                embed = discord.Embed(
                    title="Model Learning Progress",
                    description=f"The model has gone through {learning_history.get('learning_iterations', 0)} learning iterations",
                    color=discord.Color.purple(),
                    timestamp=datetime.now()
                )
                
                # Add accuracy info if available
                if learning_history.get('accuracy_over_time'):
                    accuracies = [entry['accuracy'] for entry in learning_history['accuracy_over_time']]
                    latest_accuracy = accuracies[-1] if accuracies else 0
                    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                    
                    accuracy_trend = "▲ Improving" if len(accuracies) >= 2 and accuracies[-1] > accuracies[-2] else "▼ Declining" if len(accuracies) >= 2 and accuracies[-1] < accuracies[-2] else "◆ Stable"
                    
                    embed.add_field(
                        name="Accuracy",
                        value=f"Latest: {latest_accuracy*100:.1f}%\nAverage: {avg_accuracy*100:.1f}%\nTrend: {accuracy_trend}",
                        inline=True
                    )
                
                # Add feature importance info
                if learning_history.get('feature_importance_over_time'):
                    latest_importance = learning_history['feature_importance_over_time'][-1]['importance'] if learning_history['feature_importance_over_time'] else []
                    
                    if latest_importance:
                        top_features = "\n".join([f"{item['feature']}: {item['importance']:.4f}" for item in latest_importance[:3]])
                        
                        embed.add_field(
                            name="Top Features",
                            value=top_features,
                            inline=True
                        )
                
                # Add retraining info
                if learning_history.get('retrain_events'):
                    latest_retrain = learning_history['retrain_events'][-1]
                    retrain_time = datetime.fromisoformat(latest_retrain['timestamp'])
                    days_ago = (datetime.now() - retrain_time).days
                    
                    embed.add_field(
                        name="Last Retrain",
                        value=f"{days_ago} days ago\nAccuracy: {latest_retrain['metrics']['accuracy']*100:.1f}%\nSamples: {latest_retrain['samples_used']}",
                        inline=True
                    )
                
                # Add trust scores for indicators
                indicator_trust = ""
                for indicator, data in self.prediction_system.ml_model.indicator_performance.items():
                    if 'trust_score' in data:
                        indicator_trust += f"{indicator}: {data['trust_score']*100:.1f}%\n"
                
                if indicator_trust:
                    embed.add_field(
                        name="Indicator Trust Scores",
                        value=indicator_trust,
                        inline=False
                    )
                
                # Generate model performance chart
                chart_buffer = await self.generate_learning_chart()
                
                if chart_buffer:
                    await ctx.send(embed=embed, file=discord.File(chart_buffer, 'learning.png'))
                else:
                    await ctx.send(embed=embed)
                
            except Exception as e:
                await ctx.send(f"Error fetching learning progress: {str(e)}")
        
        @self.bot.command(name="status", help="Check the system status")
        async def status(ctx):
            """Show system status information"""
            try:
                # Get logs directory info
                log_files = os.listdir('logs')
                prediction_files = [f for f in log_files if f.startswith('predictions_')]
                most_recent = max(prediction_files, key=lambda x: os.path.getmtime(os.path.join('logs', x))) if prediction_files else "None"
                
                # Count total predictions
                total_predictions = 0
                for file in prediction_files:
                    try:
                        with open(os.path.join('logs', file), 'r') as f:
                            predictions = json.load(f)
                            total_predictions += len(predictions)
                    except:
                        pass
                
                # Create embed
                embed = discord.Embed(
                    title="Crypto Prediction System Status",
                    color=discord.Color.blue(),
                    timestamp=datetime.now()
                )
                
                embed.add_field(name="System Status", value="Online ✅", inline=False)
                embed.add_field(name="Total Predictions", value=str(total_predictions), inline=True)
                embed.add_field(name="Most Recent Log", value=most_recent, inline=True)
                embed.add_field(name="Alert Channels", value=str(len(self.alert_channels)), inline=True)
                
                await ctx.send(embed=embed)
                
            except Exception as e:
                await ctx.send(f"Error checking status: {str(e)}")
        
        @self.bot.command(name="alerts", help="Configure alerts for this channel")
        async def alerts(ctx, min_confidence: float = None):
            """Configure alerts for the current channel"""
            channel_id = ctx.channel.id
            
            if min_confidence is None:
                # Show current setting
                if channel_id in self.alert_channels:
                    await ctx.send(f"Alerts for this channel are set with minimum confidence of {self.alert_channels[channel_id]*100:.0f}%")
                else:
                    await ctx.send("Alerts are not enabled for this channel. Use `!crypto alerts 0.7` to enable with 70% confidence.")
                return
            
            if min_confidence <= 0:
                # Disable alerts
                if channel_id in self.alert_channels:
                    del self.alert_channels[channel_id]
                    self.save_alert_channels()
                    await ctx.send("Alerts disabled for this channel.")
                else:
                    await ctx.send("Alerts were already disabled for this channel.")
            else:
                # Enable alerts with specified confidence
                min_confidence = min(1.0, max(0.1, min_confidence))  # Clamp between 0.1 and 1.0
                self.alert_channels[channel_id] = min_confidence
                self.save_alert_channels()
                await ctx.send(f"Alerts enabled for this channel with minimum confidence of {min_confidence*100:.0f}%")
        
        @self.bot.command(name="history", help="Show prediction history")
        async def history(ctx, days: int = 7):
            """Show prediction history as a chart"""
            await ctx.send(f"Generating prediction history for the past {days} days...")
            
            try:
                # Get prediction history
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                all_predictions = []
                
                # Load prediction logs
                for i in range(days + 1):
                    date = end_date - timedelta(days=i)
                    log_file = f"logs/predictions_{date.strftime('%Y%m%d')}.json"
                    
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            try:
                                predictions = json.load(f)
                                all_predictions.extend(predictions)
                            except json.JSONDecodeError:
                                continue
                
                if not all_predictions:
                    await ctx.send("No predictions found for the specified period.")
                    return
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame(all_predictions)
                
                # Parse timestamps
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Create plot
                plt.figure(figsize=(10, 6))
                
                # Create a numerical signal column for plotting
                df['signal_num'] = df['signal'].map({'buy': 1, 'neutral': 0, 'sell': -1})
                
                # Plot signal and confidence
                plt.plot(df['timestamp'], df['signal_num'], 'o-', label='Signal')
                
                # Color-code points by signal
                colors = df['signal'].map({'buy': 'green', 'neutral': 'blue', 'sell': 'red'})
                plt.scatter(df['timestamp'], df['signal_num'], c=colors, s=df['confidence']*100, alpha=0.7)
                
                # Add lines at buy, neutral, and sell levels
                plt.axhline(y=1, color='green', linestyle='--', alpha=0.3)
                plt.axhline(y=0, color='blue', linestyle='--', alpha=0.3)
                plt.axhline(y=-1, color='red', linestyle='--', alpha=0.3)
                
                plt.ylabel('Signal')
                plt.yticks([-1, 0, 1], ['Sell', 'Neutral', 'Buy'])
                plt.title(f'Prediction History (Last {days} days)')
                plt.grid(True, alpha=0.3)
                
                # Save figure to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Send the image
                await ctx.send(file=discord.File(buf, 'prediction_history.png'))
                
            except Exception as e:
                await ctx.send(f"Error generating history: {str(e)}")
        
        @self.bot.command(name="chk", help="Show detailed system status and model information")
        async def check_status(ctx):
            """Show detailed system information and status"""
            try:
                # Create a rich embed for system status
                embed = discord.Embed(
                    title="Crypto System Detailed Status",
                    color=discord.Color.gold(),
                    timestamp=datetime.now()
                )
                
                # Add basic system info
                embed.add_field(
                    name="System Components", 
                    value=(
                        f"Indicators: {'✅' if self.prediction_system.indicators else '❌'}\n"
                        f"ML Model: {'✅' if self.prediction_system.ml_model else '❌'}\n"
                        f"News Analyzer: {'✅' if self.prediction_system.news_analyzer else '❌'}"
                    ),
                    inline=False
                )
                
                # Get logs directory info
                log_files = os.listdir('logs')
                prediction_files = [f for f in log_files if f.startswith('predictions_')]
                most_recent = max(prediction_files, key=lambda x: os.path.getmtime(os.path.join('logs', x))) if prediction_files else "None"
                
                # Count total predictions
                total_predictions = 0
                for file in prediction_files:
                    try:
                        with open(os.path.join('logs', file), 'r') as f:
                            predictions = json.load(f)
                            total_predictions += len(predictions)
                    except:
                        pass
                
                embed.add_field(name="Total Predictions", value=str(total_predictions), inline=True)
                embed.add_field(name="Most Recent Log", value=most_recent, inline=True)
                
                # ML Model details
                if self.prediction_system.ml_model:
                    ml = self.prediction_system.ml_model
                    
                    # Get model info
                    model_info = f"Iterations: {ml.learning_history.get('learning_iterations', 0)}"
                    
                    if hasattr(ml, 'last_retrain_time'):
                        days_since = (datetime.now() - ml.last_retrain_time).days
                        model_info += f"\nLast retrain: {days_since} days ago"
                    
                    if ml.learning_history.get('accuracy_over_time'):
                        latest_acc = ml.learning_history['accuracy_over_time'][-1]['accuracy']
                        model_info += f"\nAccuracy: {latest_acc*100:.1f}%"
                    
                    embed.add_field(name="ML Model", value=model_info, inline=False)
                    
                    # If there are top features, add them
                    if ml.learning_history.get('feature_importance_over_time'):
                        latest = ml.learning_history['feature_importance_over_time'][-1]
                        top_features = "\n".join([
                            f"{i+1}. {item['feature']} ({item['importance']:.4f})"
                            for i, item in enumerate(latest['importance'][:3])
                        ])
                        
                        if top_features:
                            embed.add_field(name="Top Features", value=top_features, inline=True)
                
                # Indicators details
                if self.prediction_system.indicators:
                    indicators = self.prediction_system.indicators
                    
                    # Get indicators performance
                    perf = {}
                    for k, v in indicators.indicator_performance.items():
                        correct = v.get('correct', 0)
                        incorrect = v.get('incorrect', 0)
                        total = correct + incorrect
                        accuracy = correct / total if total > 0 else 0.5
                        perf[k] = {
                            'acc': accuracy,
                            'weight': v.get('weight', 1.0),
                            'total': total
                        }
                    
                    # Format top 3 indicators by accuracy
                    top_indicators = sorted(perf.items(), key=lambda x: x[1]['acc'], reverse=True)[:3]
                    indicator_info = "\n".join([
                        f"{name}: {data['acc']*100:.1f}% acc, weight: {data['weight']:.1f}"
                        for name, data in top_indicators if data['total'] > 0
                    ])
                    
                    embed.add_field(name="Top Indicators", value=indicator_info or "No indicator data", inline=True)
                    
                    # Current weights
                    weights_info = "\n".join([
                        f"{k}: {v:.2f}" for k, v in indicators.indicator_weights.items()
                    ][:5])  # Show top 5 weights
                    
                    embed.add_field(name="Indicator Weights", value=weights_info, inline=True)
                
                # Memory usage
                try:
                    import psutil
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / 1024**2  # MB
                    embed.add_field(name="Memory Usage", value=f"{memory_usage:.1f} MB", inline=True)
                except:
                    pass
                
                # Add latest prediction if available
                today = datetime.now().strftime('%Y%m%d')
                log_file = f"logs/predictions_{today}.json"
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            predictions = json.load(f)
                            if predictions:
                                latest = predictions[-1]
                                
                                latest_info = (
                                    f"Signal: {latest.get('signal', 'neutral').upper()}\n"
                                    f"Confidence: {latest.get('confidence', 0)*100:.1f}%\n"
                                    f"Time: {datetime.fromisoformat(latest.get('timestamp', '')).strftime('%H:%M:%S')}"
                                )
                                
                                embed.add_field(name="Latest Prediction", value=latest_info, inline=False)
                    except:
                        pass
                
                await ctx.send(embed=embed)
                
                # Generate and send a detailed status image
                status_img = await self._generate_system_status_chart()
                if status_img:
                    await ctx.send(file=discord.File(status_img, 'system_status.png'))
                
            except Exception as e:
                await ctx.send(f"Error generating system status: {str(e)}")
    
    async def generate_forecast_chart(self, forecast_data):
        """Generate a chart visualizing the forecast"""
        try:
            forecasts = forecast_data.get('forecasts', [])
            if not forecasts:
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract data for plotting
            days = [d['day'] for d in forecasts]
            confidences = [d['confidence'] for d in forecasts]
            signals = [d['prediction'] for d in forecasts]
            
            # Create a line connecting the days
            ax.plot(days, range(len(days)), 'k--', alpha=0.3)
            
            # Plot each day's prediction
            for i, day in enumerate(days):
                y = i
                color = 'green' if signals[i] == 'buy' else 'red'
                size = 100 + (confidences[i] * 200)  # Bigger circle for higher confidence
                
                ax.scatter(day, y, c=color, s=size, alpha=0.7)
                ax.text(day + 0.1, y, f"{signals[i].upper()} ({confidences[i]*100:.0f}%)", 
                       va='center', fontweight='bold', color=color)
            
            # Set y-axis labels to dates
            ax.set_yticks(range(len(days)))
            ax.set_yticklabels([d['date'] for d in forecasts])
            
            # Set x-axis limit slightly larger than the day range
            ax.set_xlim(0.5, max(days) + 0.5)
            
            # Set title and labels
            ax.set_title(f"{len(days)}-Day Forecast")
            ax.set_xlabel("Days Ahead")
            
            # Add a disclaimer
            fig.text(0.5, 0.01, "For educational purposes only. Not financial advice.", 
                    ha='center', fontsize=8, style='italic')
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            print(f"Error generating forecast chart: {e}")
            return None
    
    async def generate_learning_chart(self):
        """Generate a chart showing model learning progress"""
        try:
            # Check if ML model is available
            if not self.prediction_system.ml_model:
                return None
                
            # Use the model's chart generation method
            chart_buffer = self.prediction_system.ml_model._generate_performance_chart()
            return chart_buffer
            
        except Exception as e:
            print(f"Error generating learning chart: {e}")
            return None
    
    async def _generate_system_status_chart(self):
        """Generate a chart showing system status and statistics"""
        try:
            # Create a figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            # Get prediction history data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            all_predictions = []
            
            # Load prediction logs
            for i in range(7):
                date = end_date - timedelta(days=i)
                log_file = f"logs/predictions_{date.strftime('%Y%m%d')}.json"
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        try:
                            predictions = json.load(f)
                            all_predictions.extend(predictions)
                        except:
                            continue
            
            if all_predictions:
                # Convert to DataFrame for analysis
                import pandas as pd
                df = pd.DataFrame(all_predictions)
                
                # Parse timestamps
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Create a numerical signal column for plotting
                df['signal_num'] = df['signal'].map({'buy': 1, 'neutral': 0, 'sell': -1})
                
                # Plot signal and confidence
                axs[0].plot(df['timestamp'], df['signal_num'], 'o-', label='Signal')
                
                # Color-code points by signal
                colors = df['signal'].map({'buy': 'green', 'neutral': 'blue', 'sell': 'red'})
                axs[0].scatter(df['timestamp'], df['signal_num'], c=colors, s=df['confidence']*100, alpha=0.7)
                
                # Add lines at buy, neutral, and sell levels
                axs[0].axhline(y=1, color='green', linestyle='--', alpha=0.3)
                axs[0].axhline(y=0, color='blue', linestyle='--', alpha=0.3)
                axs[0].axhline(y=-1, color='red', linestyle='--', alpha=0.3)
                
                axs[0].set_ylabel('Signal')
                axs[0].set_yticks([-1, 0, 1])
                axs[0].set_yticklabels(['Sell', 'Neutral', 'Buy'])
                axs[0].set_title('Recent Prediction Signals')
                axs[0].grid(True, alpha=0.3)
            
            # Plot ML model accuracy if available
            if self.prediction_system.ml_model and self.prediction_system.ml_model.learning_history.get('accuracy_over_time'):
                accuracy_data = self.prediction_system.ml_model.learning_history['accuracy_over_time']
                
                if accuracy_data:
                    timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in accuracy_data]
                    accuracies = [entry['accuracy'] for entry in accuracy_data]
                    
                    # Plot accuracy trend
                    axs[1].plot(timestamps, accuracies, 'o-', color='purple')
                    axs[1].set_title('ML Model Accuracy Over Time')
                    axs[1].set_ylabel('Accuracy')
                    axs[1].set_ylim([0, 1])
                    axs[1].grid(True, alpha=0.3)
                    
                    # Highlight retraining events
                    if self.prediction_system.ml_model.learning_history.get('retrain_events'):
                        retrain_times = [datetime.fromisoformat(entry['timestamp']) 
                                         for entry in self.prediction_system.ml_model.learning_history['retrain_events']]
                        
                        for rt in retrain_times:
                            axs[1].axvline(x=rt, color='red', linestyle='--', alpha=0.3)
            else:
                # If no ML data, show indicator performance
                if self.prediction_system.indicators and hasattr(self.prediction_system.indicators, 'indicator_performance'):
                    perf = self.prediction_system.indicators.indicator_performance
                    indicators = []
                    accuracies = []
                    
                    for k, v in perf.items():
                        correct = v.get('correct', 0)
                        incorrect = v.get('incorrect', 0)
                        total = correct + incorrect
                        if total > 0:
                            indicators.append(k)
                            accuracies.append(correct / total)
                    
                    if indicators:
                        # Only show top indicators if there are many
                        if len(indicators) > 8:
                            # Sort by accuracy and take top 8
                            sorted_data = sorted(zip(indicators, accuracies), key=lambda x: x[1], reverse=True)
                            indicators = [x[0] for x in sorted_data[:8]]
                            accuracies = [x[1] for x in sorted_data[:8]]
                        
                        axs[1].bar(indicators, accuracies, color='skyblue')
                        axs[1].set_title('Indicator Performance')
                        axs[1].set_ylabel('Accuracy')
                        axs[1].set_ylim([0, 1])
                        axs[1].grid(True, alpha=0.3)
                        plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save figure to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            print(f"Error generating system status chart: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_color_for_signal(self, signal):
        """Return Discord color based on signal"""
        if signal.lower() == 'buy':
            return discord.Color.green()
        elif signal.lower() == 'sell':
            return discord.Color.red()
        else:
            return discord.Color.blue()
    
    @tasks.loop(minutes=5)
    async def check_for_predictions(self):
        """Check for new predictions and send alerts"""
        if not self.alert_channels:
            return
            
        try:
            # Get latest prediction file
            today = datetime.now().strftime('%Y%m%d')
            log_file = f"logs/predictions_{today}.json"
            
            if not os.path.exists(log_file):
                return
                
            with open(log_file, 'r') as f:
                predictions = json.load(f)
                
            if not predictions:
                return
                
            # Get the latest prediction
            latest = predictions[-1]
            
            # Check if it's less than 15 minutes old
            timestamp = datetime.fromisoformat(latest['timestamp'])
            age = datetime.now() - timestamp
            
            if age > timedelta(minutes=15):
                return  # Too old
                
            # Check confidence threshold and send alerts
            confidence = latest.get('confidence', 0)
            signal = latest.get('signal', 'neutral')
            
            # Only alert for buy/sell signals
            if signal == 'neutral':
                return
                
            for channel_id, min_confidence in self.alert_channels.items():
                if confidence >= min_confidence:
                    # Send alert to this channel
                    channel = self.bot.get_channel(channel_id)
                    if channel:
                        # Create embed
                        embed = discord.Embed(
                            title=f"🚨 {signal.upper()} Alert!",
                            description=f"New high-confidence {signal} signal detected",
                            color=self.get_color_for_signal(signal),
                            timestamp=timestamp
                        )
                        
                        embed.add_field(
                            name="Signal", 
                            value=signal.upper(), 
                            inline=True
                        )
                        embed.add_field(
                            name="Confidence", 
                            value=f"{confidence*100:.1f}%", 
                            inline=True
                        )
                        embed.add_field(
                            name="Price", 
                            value=f"${latest.get('price', 0):,.2f}", 
                            inline=True
                        )
                        
                        await channel.send(embed=embed)
                        
        except Exception as e:
            print(f"Error checking for predictions: {e}")
    
    @tasks.loop(minutes=60)
    async def check_for_learning_updates(self):
        """Check if the model needs retraining based on feedback"""
        try:
            if not self.prediction_system.ml_model:
                return
                
            # Check if we have enough feedback samples to consider retraining
            feedback_count = len(self.prediction_system.ml_model.feedback_samples)
            min_required = self.prediction_system.ml_model.min_feedback_samples
            
            if feedback_count >= min_required:
                # Log this to the learning channel
                learning_channel = self.bot.get_channel(self.learning_channel_id)
                if learning_channel:
                    await learning_channel.send(
                        f"🧠 Model has collected {feedback_count} feedback samples " +
                        f"(minimum {min_required} required). Evaluating if retraining is needed..."
                    )
                
                # Check if retraining is needed
                days_since_retrain = (datetime.now() - self.prediction_system.ml_model.last_retrain_time).days
                
                if days_since_retrain >= self.prediction_system.ml_model.auto_retrain_days:
                    if learning_channel:
                        await learning_channel.send(
                            f"⏰ Scheduled retraining triggered - {days_since_retrain} days since last retrain"
                        )
                    
                    # Trigger retraining in a background task to avoid blocking the bot
                    asyncio.create_task(self._trigger_model_retraining())
        except Exception as e:
            print(f"Error checking for learning updates: {e}")
    
    async def _trigger_model_retraining(self):
        """Trigger model retraining and report results"""
        try:
            # This runs as a background task to avoid blocking the bot
            # Call the model's retrain method
            success = self.prediction_system.ml_model._evaluate_retraining_need()
            
            # Log the result to the learning channel
            learning_channel = self.bot.get_channel(self.learning_channel_id)
            if learning_channel:
                if success:
                    await learning_channel.send("✅ Model retraining completed successfully!")
                else:
                    await learning_channel.send("ℹ️ Model retraining was not needed at this time.")
        except Exception as e:
            print(f"Error during model retraining: {e}")
            
            # Log the error to the learning channel
            learning_channel = self.bot.get_channel(self.learning_channel_id)
            if learning_channel:
                await learning_channel.send(f"❌ Error during model retraining: {str(e)}")
    
    def load_alert_channels(self):
        """Load alert channels configuration from file"""
        config_path = 'data/discord_alerts.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.alert_channels = json.load(f)
            except:
                self.alert_channels = {}
    
    def save_alert_channels(self):
        """Save alert channels configuration to file"""
        os.makedirs('data', exist_ok=True)
        config_path = 'data/discord_alerts.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(self.alert_channels, f)
        except Exception as e:
            print(f"Error saving alert channels: {e}")
    
    async def start(self):
        """Start the Discord bot"""
        if not self.token:
            print("Cannot start Discord bot: No token provided")
            return
            
        try:
            await self.bot.start(self.token)
        except Exception as e:
            print(f"Error starting Discord bot: {e}")
    
    def run(self):
        """Run the bot (blocking)"""
        if not self.token:
            print("Cannot start Discord bot: No token provided")
            return
            
        try:
            self.bot.run(self.token)
        except Exception as e:
            print(f"Error running Discord bot: {e}")
            
    def start_async(self):
        """Start the bot in a non-blocking manner"""
        asyncio.create_task(self.start())
