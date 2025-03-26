import pandas as pd
import numpy as np
import os
import json
import pickle
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import io
from learning_tracker import LearningTracker

class CryptoMLModel:
    def __init__(self, discord_logger=None):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.discord_logger = discord_logger
        
        # Replace the learning history tracking with LearningTracker
        self.learning_tracker = LearningTracker(self.model_dir, discord_logger)
        self.learning_history = self.learning_tracker.history
        
        # All possible features - the model will adapt based on what's available
        self.all_feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'ema_20', 'ema_50', 'ema_200',
            'bb_percent_b', 'bb_bandwidth',
            'stoch_k', 'stoch_d',
            'ichimoku_senkou_a', 'ichimoku_senkou_b',
            'obv', 'obv_ema',
            'atr', 'atr_percent',
            'volume', 'volume_sma_20', 'volume_pct_change',
            'price_change_1h', 'price_change_24h', 'volatility',
            'sentiment_score'
        ]
        
        # Core features that are most important
        self.core_feature_columns = [
            'rsi', 'macd', 'ema_50', 'atr', 'volume_pct_change'
        ]
        
        # Dynamic features actually used (will be set based on available data)
        self.feature_columns = []
        
        # Indicator performance tracking
        self.indicator_performance = {
            'rsi': {'correct': 0, 'incorrect': 0, 'weight': 1.0, 'trust_score': 0.5},
            'macd': {'correct': 0, 'incorrect': 0, 'weight': 1.0, 'trust_score': 0.5},
            'ema': {'correct': 0, 'incorrect': 0, 'weight': 1.0, 'trust_score': 0.5},
            'volume': {'correct': 0, 'incorrect': 0, 'weight': 1.0, 'trust_score': 0.5},
            'atr': {'correct': 0, 'incorrect': 0, 'weight': 1.0, 'trust_score': 0.5},
            'bollinger': {'correct': 0, 'incorrect': 0, 'weight': 0.8, 'trust_score': 0.5},
            'stochastic': {'correct': 0, 'incorrect': 0, 'weight': 0.8, 'trust_score': 0.5},
            'ichimoku': {'correct': 0, 'incorrect': 0, 'weight': 0.7, 'trust_score': 0.5},
            'obv': {'correct': 0, 'incorrect': 0, 'weight': 0.7, 'trust_score': 0.5},
            'parabolic_sar': {'correct': 0, 'incorrect': 0, 'weight': 0.6, 'trust_score': 0.5},
            'sentiment': {'correct': 0, 'incorrect': 0, 'weight': 0.8, 'trust_score': 0.5}
        }
        
        # Self-learning parameters
        self.learning_rate = 0.1
        self.min_feedback_samples = 5  # Minimum number of samples before retraining
        self.feedback_samples = []
        self.anomaly_cases = []
        self.last_retrain_time = datetime.now()
        self.auto_retrain_days = 7  # Automatically retrain every 7 days
        
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs('data/feedback', exist_ok=True)
        
        # Load previous learning history if available
        self._load_learning_history()
    
    def log(self, message, level="INFO", send_to_discord=False):
        """Log a message and optionally send to Discord"""
        print(f"[ML Model] {level}: {message}")
        
        if send_to_discord and self.discord_logger:
            self.discord_logger.send_log(message, level)
    
    def _load_learning_history(self):
        """Load previous learning history from file"""
        # The tracker already loads history when initialized
        self.learning_history = self.learning_tracker.history
    
    def _save_learning_history(self):
        """Save learning history to file"""
        return self.learning_tracker.save_history()
            
    def load_historical_data(self, file_path=None):
        """Load historical data from CSV or fetch from APIs"""
        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
        
        # Try to fetch real historical data
        try:
            return self._fetch_real_historical_data()
        except Exception as e:
            print(f"Error fetching real historical data: {e}")
            print("Falling back to synthetic data generation")
            
            # Generate synthetic data as fallback
            return self._generate_synthetic_data()
    
    def _fetch_real_historical_data(self):
        """Fetch real historical data from Binance, CoinGecko, or others"""
        print("Fetching real historical Bitcoin price data...")
        
        # Try CoinGecko API first (doesn't require API key)
        try:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': 365,  # One year of data
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise Exception(f"CoinGecko API error: {response.status_code}")
                
            data = response.json()
            
            # Extract price and volume data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                raise Exception("No price data returned from CoinGecko")
                
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                df = pd.merge(df, volume_df, on='timestamp', how='left')
            else:
                df['volume'] = 0
            
            # Calculate price changes and returns
            df['price_change_24h'] = df['price'].pct_change(1)
            df['price_change_7d'] = df['price'].pct_change(7)
            
            # Create target variable: 1 if price increases next day
            df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
            
            # Add basic technical indicators
            df['sma_7'] = df['price'].rolling(window=7).mean()
            df['sma_30'] = df['price'].rolling(window=30).mean()
            
            # Calculate volatility (standard deviation of returns)
            df['volatility'] = df['price_change_24h'].rolling(window=7).std()
            
            # Drop rows with NaN values
            df = df.dropna()
            
            print(f"Successfully fetched {len(df)} days of Bitcoin historical data")
            return df
            
        except Exception as e:
            print(f"Error with CoinGecko API: {e}")
            raise
    
    def _generate_synthetic_data(self, rows=365):
        """Generate synthetic data for testing when no real data is available"""
        print("Generating synthetic data for model training...")
        
        # Create dates for the past year with daily data
        dates = pd.date_range(
            end=datetime.now(), 
            periods=rows,
            freq='D'
        )
        
        # Start price around 30,000
        base_price = 30000
        
        # Generate random price movement with realistic properties
        np.random.seed(42)  # For reproducibility
        
        # Create random walk for price with volatility clustering
        returns = np.random.normal(0.0002, 0.02, size=rows)  # Small positive drift
        
        # Add volatility clustering (GARCH-like effect)
        volatility = np.ones(rows) * 0.02
        for i in range(1, rows):
            volatility[i] = 0.9 * volatility[i-1] + 0.1 * abs(returns[i-1])
            returns[i] = np.random.normal(0.0002, volatility[i])
        
        # Convert returns to price
        price = base_price * np.cumprod(1 + returns)
        
        # Calculate various features
        rsi_values = np.clip(50 + returns * 500, 0, 100)  # Approximate RSI
        macd_values = np.convolve(returns, np.ones(12)/12, mode='same') - np.convolve(returns, np.ones(26)/26, mode='same')
        volume = np.random.normal(10000, 3000, size=rows) * (1 + np.abs(returns)*10)  # Volume correlates with volatility
        
        # Create dataframe
        df = pd.DataFrame({
            'timestamp': dates,
            'price': price,
            'volume': volume,
            'returns': returns,
            'rsi': rsi_values,
            'macd': macd_values,
            'ema_50': np.convolve(price, np.ones(50)/50, mode='same'),
            'atr': np.abs(np.random.normal(0, 1, size=rows)) * price * 0.02,
            'volatility': volatility,
            'volume_pct_change': np.random.normal(0, 0.1, size=rows),
            'price_change_24h': returns,
            'price_change_7d': np.convolve(returns, np.ones(7), mode='same'),
            'sentiment_score': np.random.normal(0, 0.3, size=rows),  # Random sentiment
        })
        
        # Create binary target (1 if price goes up next day)
        df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
        
        # Drop last row (no target)
        df = df[:-1]
        
        return df
    
    def prepare_feature_data(self, data):
        """Prepare features for model training, adapting to available data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        # Identify which features are available in the data
        available_columns = [col for col in self.all_feature_columns if col in data.columns]
        
        # Always include core features that can be calculated if missing
        for col in self.core_feature_columns:
            if col not in available_columns and col in data.columns:
                available_columns.append(col)
        
        # If too few features are available, raise an error
        if len(available_columns) < 3:
            raise ValueError(f"Not enough feature columns found. Required at least 3, Found: {available_columns}")
        
        # Set the actual features to be used
        self.feature_columns = available_columns
        print(f"Using features: {self.feature_columns}")
        
        X = data[available_columns]
        
        # If target column is in data
        if 'target' in data.columns:
            y = data['target']
        else:
            y = None
        
        return X, y
    
    def train_model(self, data=None, file_path=None, is_retraining=False):
        """Train the ML model on historical data"""
        if data is None:
            data = self.load_historical_data(file_path)
        
        X, y = self.prepare_feature_data(data)
        
        if y is None:
            raise ValueError("Target variable not found in data")
        
        # Record start time for performance tracking
        start_time = datetime.now()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle missing values with imputer
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        # Initialize and train model
        training_message = "Retraining" if is_retraining else "Training new"
        self.log(f"{training_message} model with gradient boosting classifier...", send_to_discord=True)
        
        # Use previous model parameters if retraining for continuity
        if is_retraining and self.model:
            old_params = self.model.get_params()
            # Keep some parameters but allow for adjustments in others
            params = {
                'n_estimators': min(old_params['n_estimators'] + 20, 200),  # Add more trees up to a limit
                'learning_rate': max(old_params['learning_rate'] * 0.9, 0.01),  # Gradually decrease learning rate
                'max_depth': old_params['max_depth'],
                'random_state': 42
            }
            self.model = GradientBoostingClassifier(**params)
        else:
            # New model with default parameters
            self.model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=4,
                random_state=42
            )
            
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importances
        feature_importance = []
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            importance_msg = "Feature importance:\n"
            for feature, importance in feature_importance:
                importance_msg += f"  {feature}: {importance:.4f}\n"
            
            self.log(importance_msg)
        
        # Record training result in learning history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'is_retraining': is_retraining,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            },
            'feature_importance': [{"feature": f, "importance": float(i)} for f, i in feature_importance],
            'samples_used': len(X),
            'training_duration_seconds': (datetime.now() - start_time).total_seconds()
        }
        
        self.learning_history['retrain_events'].append(training_record)
        self.learning_history['accuracy_over_time'].append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy)
        })
        self.learning_history['feature_importance_over_time'].append({
            'timestamp': datetime.now().isoformat(),
            'importance': [{"feature": f, "importance": float(i)} for f, i in feature_importance[:3]]  # Top 3
        })
        self.learning_history['learning_iterations'] += 1
        
        self._save_learning_history()
        
        # Generate a chart of model performance
        if self.discord_logger:
            self._generate_performance_chart()
        
        # Log results to Discord
        result_msg = (
            f"Model {training_message.lower()} complete!\n"
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}, F1: {f1:.4f}\n"
            f"Top features: {', '.join([f[0] for f in feature_importance[:3]])}"
        )
        self.log(result_msg, send_to_discord=True)
        
        # Update last retrain time
        self.last_retrain_time = datetime.now()
        
        # Save model
        self.save_model()
        
        # After training is complete, update the learning tracker
        training_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        self.learning_tracker.add_training_result(
            metrics=training_metrics,
            feature_importance=[{"feature": f, "importance": float(i)} for f, i in feature_importance],
            samples_used=len(X),
            is_retraining=is_retraining
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_columns': self.feature_columns,
            'confusion_matrix': cm.tolist()
        }
    
    def predict(self, indicator_data):
        """Make predictions using the trained model, adapting to available data"""
        if self.model is None:
            # Try to load a saved model
            self.load_model()
            
            # If still no model, train a new one
            if self.model is None:
                print("No trained model found. Training a new model...")
                self.train_model()
        
        # Prepare features from indicator data
        features = {}
        
        # Extract whatever features are available
        indicators = indicator_data.get('indicators', {})
        for feature in self.all_feature_columns:
            # Direct mapping for common indicators
            if feature in indicators:
                features[feature] = indicators[feature]
            # Special handling for specific features
            elif feature == 'rsi':
                features[feature] = indicators.get('rsi', 50)
            elif feature == 'macd':
                features[feature] = indicators.get('macd', 0)
            elif feature == 'ema_50':
                features[feature] = indicators.get('ema', indicator_data.get('price', 0))
            elif feature == 'atr':
                features[feature] = indicators.get('atr', 0)
            elif feature == 'volume_pct_change':
                features[feature] = indicators.get('volume_change_pct', 0)
            elif feature == 'volatility':
                features[feature] = indicator_data.get('volatility', 0)
            elif feature == 'sentiment_score':
                # Try to get sentiment data if available
                sentiment = indicator_data.get('components', {}).get('sentiment', {})
                features[feature] = sentiment.get('score', 0)
        
        # Create DataFrame from features
        df = pd.DataFrame([features])
        
        # Get intersection of available features and model features
        available_features = list(set(df.columns) & set(self.feature_columns))
        
        # Ensure at least 3 features are available
        if len(available_features) < 3:
            print(f"WARNING: Too few features available ({len(available_features)}). Adding synthetic features.")
            # Add synthetic features if needed
            if 'rsi' not in df.columns:
                df['rsi'] = 50  # Neutral RSI
            if 'macd' not in df.columns:
                df['macd'] = 0  # Neutral MACD
            if 'volatility' not in df.columns:
                df['volatility'] = 0.01  # Low volatility
                
            # Update available features
            available_features = list(set(df.columns) & set(self.feature_columns))
        
        # Create feature array with only available features
        X = df[available_features]
        
        # Handle missing values
        X_imputed = self.imputer.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        # Make prediction
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]
        
        # Calculate confidence based on how many features were available
        feature_ratio = len(available_features) / len(self.feature_columns)
        confidence_modifier = min(1.0, 0.5 + 0.5 * feature_ratio)
        
        confidence = float(prediction_proba[prediction]) * confidence_modifier
        
        # Only return high confidence if we have solid data
        if feature_ratio < 0.6:
            confidence = min(confidence, 0.75)  # Cap confidence when missing too many features
        
        return {
            'prediction': 'buy' if prediction == 1 else 'sell',
            'confidence': confidence,
            'buy_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0,
            'sell_probability': float(prediction_proba[0]) if len(prediction_proba) > 0 else 0,
            'features_available': len(available_features),
            'features_total': len(self.feature_columns),
            'feature_coverage': feature_ratio
        }
    
    def _generate_performance_chart(self):
        """Generate a chart showing model performance over time"""
        return self.learning_tracker.generate_learning_chart()
    
    def add_feedback(self, prediction_data, actual_outcome, human_feedback=None):
        """Add feedback for a prediction to improve the model"""
        try:
            # Record the prediction and outcome
            feedback = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction_data,
                'actual_outcome': actual_outcome,
                'human_feedback': human_feedback
            }
            
            self.feedback_samples.append(feedback)
            
            # Add to learning tracker
            self.learning_tracker.add_human_feedback(prediction_data, actual_outcome, human_feedback)
            
            # Save feedback to disk
            feedback_file = f"data/feedback/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback, f, indent=2)
            
            self.log(f"Recorded prediction feedback (outcome: {'correct' if prediction_data['prediction'] == actual_outcome else 'incorrect'})")
            
            # Check if we need to retrain
            if len(self.feedback_samples) >= self.min_feedback_samples:
                self.log(f"Accumulated {len(self.feedback_samples)} feedback samples, considering retraining", send_to_discord=True)
                self._evaluate_retraining_need()
            
            # Check for anomalies (cases where the model was wrong but indicators were right)
            self._check_for_anomalies(prediction_data, actual_outcome)
            
            return True
        except Exception as e:
            self.log(f"Error adding feedback: {e}", level="ERROR")
            return False
    
    def _evaluate_retraining_need(self):
        """Decide if the model should be retrained based on feedback"""
        # Count incorrect predictions
        incorrect_count = sum(1 for f in self.feedback_samples 
                            if f['prediction']['prediction'] != f['actual_outcome'])
        
        # Calculate error rate
        error_rate = incorrect_count / len(self.feedback_samples)
        
        # Check if error rate is too high or if it's been too long since last retrain
        days_since_retrain = (datetime.now() - self.last_retrain_time).days
        
        should_retrain = False
        retrain_reason = ""
        
        if error_rate > 0.3:  # More than 30% incorrect predictions
            should_retrain = True
            retrain_reason = f"high error rate ({error_rate:.2%})"
        elif days_since_retrain >= self.auto_retrain_days:
            should_retrain = True
            retrain_reason = f"scheduled retraining (last retrain: {days_since_retrain} days ago)"
        
        # Log decision
        if should_retrain:
            self.log(f"Initiating model retraining due to {retrain_reason}", send_to_discord=True)
            self._retrain_model_with_feedback()
        else:
            self.log(f"Model retraining not needed yet (error rate: {error_rate:.2%}, days since retrain: {days_since_retrain})")
            
        return should_retrain
    
    def _retrain_model_with_feedback(self):
        """Retrain the model incorporating feedback data"""
        try:
            # Get historical data as base
            base_data = self.load_historical_data()
            
            # Prepare feedback data that can be incorporated
            feedback_data = []
            
            for feedback in self.feedback_samples:
                # Extract features from the prediction data
                features = {}
                pred_data = feedback['prediction']
                
                # Extract all available features from the prediction
                for feature in self.all_feature_columns:
                    # Try to find the feature in prediction data
                    # This is complex because features might be nested in different ways
                    value = None
                    
                    # Try direct access
                    if feature in pred_data:
                        value = pred_data[feature]
                    # Try in components or indicators
                    elif 'components' in pred_data:
                        components = pred_data['components']
                        # Look in each component section
                        for comp_name, comp_data in components.items():
                            if feature in comp_data:
                                value = comp_data[feature]
                    # If still not found, try indicators
                    if value is None and 'indicators' in pred_data:
                        indicators = pred_data['indicators']
                        if feature in indicators:
                            value = indicators[feature]
                    
                    # If found, add to features
                    if value is not None:
                        features[feature] = value
                
                # Add target outcome (1 for buy, 0 for sell)
                features['target'] = 1 if feedback['actual_outcome'] == 'buy' else 0
                
                # Add to feedback data if we have enough features
                if len(features) >= 3:  # Minimum required features
                    feedback_data.append(features)
            
            if not feedback_data:
                self.log("No usable feedback data for retraining", level="WARNING")
                return False
            
            # Convert feedback data to DataFrame
            feedback_df = pd.DataFrame(feedback_data)
            
            # Combine with base data if compatible
            if set(feedback_df.columns).intersection(set(base_data.columns)) >= 3:
                combined_data = pd.concat([base_data, feedback_df], ignore_index=True)
                self.log(f"Combined {len(base_data)} historical records with {len(feedback_df)} feedback records for retraining")
            else:
                # If not enough compatible columns, use feedback data alone
                combined_data = feedback_df
                self.log(f"Using {len(feedback_df)} feedback records for retraining (not compatible with base data)")
            
            # Retrain the model
            self.train_model(data=combined_data, is_retraining=True)
            
            # Clear feedback samples after successful retraining
            self.feedback_samples = []
            
            return True
            
        except Exception as e:
            self.log(f"Error retraining model with feedback: {e}", level="ERROR", send_to_discord=True)
            return False
    
    def _check_for_anomalies(self, prediction_data, actual_outcome):
        """Identify market anomalies where indicators and model disagree"""
        try:
            # Get model prediction and indicators
            model_prediction = prediction_data.get('prediction', 'neutral')
            
            # Get indicators predictions
            indicators = prediction_data.get('components', {}).get('indicator', {})
            indicator_signal = indicators.get('signal', 'neutral')
            
            # Check if model and indicators disagree
            if model_prediction != indicator_signal:
                # Check which one was correct
                if model_prediction == actual_outcome:
                    # Model was right, indicators were wrong - possible anomaly detected by ML
                    anomaly = {
                        'timestamp': datetime.now().isoformat(),
                        'model_prediction': model_prediction,
                        'indicator_signal': indicator_signal,
                        'actual_outcome': actual_outcome,
                        'prediction_data': prediction_data
                    }
                    
                    self.anomaly_cases.append(anomaly)
                    
                    # Add to learning tracker
                    self.learning_tracker.add_anomaly(anomaly)
                    
                    # Save anomaly to disk
                    anomaly_file = f"data/feedback/anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(anomaly_file, 'w') as f:
                        json.dump(anomaly, f, indent=2)
                    
                    self.log(f"Detected market anomaly: Model correctly predicted {model_prediction} " + 
                             f"while indicators suggested {indicator_signal}", 
                             level="INFO", send_to_discord=True)
                    
                    # Analyze what the model saw that indicators missed
                    self._analyze_anomaly(anomaly)
                    
        except Exception as e:
            self.log(f"Error checking for anomalies: {e}", level="ERROR")
    
    def _analyze_anomaly(self, anomaly):
        """Analyze what the model detected that standard indicators missed"""
        try:
            # This is where we'd analyze feature importance for this specific prediction
            # For now, we'll just log that we found an anomaly
            
            # In a more advanced implementation, we could:
            # 1. Use SHAP values to understand which features contributed to this prediction
            # 2. Compare against typical feature importance
            # 3. Identify what unusual pattern the model detected
            
            self.log(f"Model detected an anomaly the indicators missed. " +
                    f"This helps build trust in the ML model's ability to find hidden patterns.", 
                    send_to_discord=True)
            
        except Exception as e:
            self.log(f"Error analyzing anomaly: {e}", level="ERROR")
    
    def update_indicator_weights(self, indicator_data, actual_outcome):
        """Update indicator weights based on performance"""
        indicators = indicator_data.get('indicators', {})
        
        # Expected direction based on indicator signals
        expected = {
            'rsi': 1 if indicators.get('rsi_signal', 0) > 0 else (-1 if indicators.get('rsi_signal', 0) < 0 else 0),
            'macd': 1 if indicators.get('macd_signal', 0) > 0 else (-1 if indicators.get('macd_signal', 0) < 0 else 0),
            'ema': 1 if indicators.get('ema_signal', 0) > 0 else (-1 if indicators.get('ema_signal', 0) < 0 else 0),
            'volume': 1 if indicators.get('volume_signal', 0) > 0 else (-1 if indicators.get('volume_signal', 0) < 0 else 0),
            'bollinger': 1 if indicators.get('bollinger_signal', 0) > 0 else (-1 if indicators.get('bollinger_signal', 0) < 0 else 0),
            'stochastic': 1 if indicators.get('stochastic_signal', 0) > 0 else (-1 if indicators.get('stochastic_signal', 0) < 0 else 0),
            'ichimoku': 1 if indicators.get('ichimoku_signal', 0) > 0 else (-1 if indicators.get('ichimoku_signal', 0) < 0 else 0),
            'parabolic_sar': 1 if indicators.get('parabolic_sar_signal', 0) > 0 else (-1 if indicators.get('parabolic_sar_signal', 0) < 0 else 0),
            'obv': 1 if indicators.get('obv_signal', 0) > 0 else (-1 if indicators.get('obv_signal', 0) < 0 else 0)
        }
        
        # Actual outcome: 1 for price increase, -1 for decrease
        actual = 1 if actual_outcome == 'buy' else -1
        
        correct_indicators = []
        incorrect_indicators = []
        
        # Update performance for each indicator
        for indicator, direction in expected.items():
            if direction == 0:  # Skip neutral signals
                continue
                
            if direction == actual:
                self.indicator_performance[indicator]['correct'] += 1
                correct_indicators.append(indicator)
            else:
                self.indicator_performance[indicator]['incorrect'] += 1
                incorrect_indicators.append(indicator)
            
            # Recalculate weight and trust score
            total = self.indicator_performance[indicator]['correct'] + self.indicator_performance[indicator]['incorrect']
            
            if total > 0:
                # Calculate accuracy-based weight
                accuracy = self.indicator_performance[indicator]['correct'] / total
                
                # Update trust score (using exponential moving average for smoothing)
                old_trust = self.indicator_performance[indicator]['trust_score']
                new_trust = 0.9 * old_trust + 0.1 * accuracy
                self.indicator_performance[indicator]['trust_score'] = new_trust
                
                # Weight is now based on trust score with a minimum value
                self.indicator_performance[indicator]['weight'] = max(0.2, new_trust)
        
        # Log which indicators were correct/incorrect
        if correct_indicators:
            self.log(f"Correct indicators: {', '.join(correct_indicators)}")
        if incorrect_indicators:
            self.log(f"Incorrect indicators: {', '.join(incorrect_indicators)}")
        
        return self.indicator_performance
    
    def get_forecast(self, symbol='BTC/USD', days=1):
        """Generate a forecast for the specified number of days ahead"""
        try:
            self.log(f"Generating {days}-day forecast for {symbol}...", send_to_discord=True)
            
            # Get the latest real data
            latest_data = self._fetch_latest_data(symbol)
            if not latest_data:
                return {"error": "Failed to fetch latest data for forecasting"}
            
            # Prepare features from real data
            features = self._prepare_forecast_features(latest_data)
            
            # Generate day-by-day predictions
            forecasts = []
            current_features = features.copy()
            
            for day in range(1, days+1):
                # Make prediction with current features
                prediction = self._forecast_day(current_features)
                
                # Add to forecasts
                forecasts.append({
                    'day': day,
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'price_direction': 'up' if prediction['prediction'] == 'buy' else 'down',
                    'features_used': list(current_features.keys())
                })
                
                # Update features based on prediction for next day
                self._update_features_for_next_day(current_features, prediction)
            
            # Create the forecast result
            result = {
                'symbol': symbol,
                'generated_at': datetime.now().isoformat(),
                'days_ahead': days,
                'current_price': latest_data.get('close', 0),
                'forecasts': forecasts,
                'model_version': self.learning_history.get('learning_iterations', 0),
                'trust_score': self._calculate_model_trust_score()
            }
            
            self.log(f"Forecast complete: {days} days ahead for {symbol}", send_to_discord=True)
            return result
            
        except Exception as e:
            self.log(f"Error generating forecast: {e}", level="ERROR", send_to_discord=True)
            return {"error": str(e)}
    
    def _fetch_latest_data(self, symbol):
        """Fetch the latest real data for a symbol"""
        try:
            # Clean symbol format
            clean_symbol = symbol.replace('/', '').lower()
            
            # Try to get data from CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/{clean_symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': 7,  # Get a week of data
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
                
            data = response.json()
            
            # Extract price and volume data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                raise Exception("No price data returned")
                
            # Get the most recent data point
            latest_price = prices[-1][1]  # [timestamp, price]
            latest_volume = volumes[-1][1] if volumes else 0
            
            # Calculate some basic indicators
            price_history = [p[1] for p in prices]
            
            # Calculate returns
            returns = []
            for i in range(1, len(price_history)):
                returns.append((price_history[i] - price_history[i-1]) / price_history[i-1])
            
            # Simple moving averages
            sma_7 = sum(price_history[-7:]) / 7 if len(price_history) >= 7 else latest_price
            
            # Volatility (standard deviation of returns)
            volatility = np.std(returns) if returns else 0
            
            # Simulate RSI
            gains = [r if r > 0 else 0 for r in returns]
            losses = [abs(r) if r < 0 else 0 for r in returns]
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.0001  # Avoid division by zero
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Return latest data
            return {
                'timestamp': datetime.now().isoformat(),
                'open': price_history[-2] if len(price_history) > 1 else latest_price,
                'high': max(price_history[-7:]) if len(price_history) >= 7 else latest_price,
                'low': min(price_history[-7:]) if len(price_history) >= 7 else latest_price,
                'close': latest_price,
                'volume': latest_volume,
                'sma_7': sma_7,
                'volatility': volatility,
                'rsi': rsi,
                'price_change_24h': returns[-1] if returns else 0,
                'price_history': price_history[-7:]  # Last 7 days
            }
            
        except Exception as e:
            self.log(f"Error fetching latest data: {e}", level="ERROR")
            return None
    
    def _prepare_forecast_features(self, data):
        """Prepare features for forecasting from latest data"""
        features = {}
        
        # Map data to features the model expects
        for feature in self.all_feature_columns:
            if feature in data:
                features[feature] = data[feature]
        
        # Derive additional features if possible
        price = data.get('close', 0)
        if 'rsi' not in features and 'rsi' in self.all_feature_columns:
            features['rsi'] = data.get('rsi', 50)  # Default to neutral if not available
            
        if 'ema_50' not in features and 'ema_50' in self.all_feature_columns:
            # Approximate EMA with latest price if not available
            features['ema_50'] = data.get('sma_7', price)
            
        if 'volatility' not in features and 'volatility' in self.all_feature_columns:
            features['volatility'] = data.get('volatility', 0.01)
            
        if 'volume' not in features and 'volume' in self.all_feature_columns:
            features['volume'] = data.get('volume', 0)
            
        # Add default values for required features if missing
        for required in self.core_feature_columns:
            if required not in features:
                # Add neutral/default values
                if 'rsi' in required:
                    features[required] = 50
                elif 'macd' in required:
                    features[required] = 0
                elif 'ema' in required:
                    features[required] = price
                elif 'volume' in required:
                    features[required] = 0
                elif 'atr' in required:
                    features[required] = price * 0.01  # 1% of price
        
        return features
    
    def _forecast_day(self, features):
        """Make a prediction for a single day with the given features"""
        # Create a DataFrame with the features
        df = pd.DataFrame([features])
        
        # Get available features that match model features
        available_features = list(set(df.columns) & set(self.feature_columns))
        
        # Handle missing values
        X = df[available_features]
        X_imputed = self.imputer.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        # Make prediction
        if self.model:
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            
            # Calculate confidence
            feature_ratio = len(available_features) / len(self.feature_columns)
            confidence_modifier = min(1.0, 0.5 + 0.5 * feature_ratio)
            confidence = float(prediction_proba[prediction]) * confidence_modifier
            
            return {
                'prediction': 'buy' if prediction == 1 else 'sell',
                'confidence': confidence,
                'buy_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0,
                'sell_probability': float(prediction_proba[0]) if len(prediction_proba) > 0 else 0
            }
        else:
            # No model available
            return {
                'prediction': 'neutral',
                'confidence': 0.5,
                'buy_probability': 0.5,
                'sell_probability': 0.5
            }
    
    def _update_features_for_next_day(self, features, prediction):
        """Update features based on the prediction for the next day"""
        # Simulate changes based on prediction
        price_change = 0.01 if prediction['prediction'] == 'buy' else -0.01  # 1% price movement
        
        # Update RSI
        if 'rsi' in features:
            if prediction['prediction'] == 'buy':
                # RSI should move up on buy signal
                features['rsi'] = min(85, features['rsi'] + 5)
            else:
                # RSI should move down on sell signal
                features['rsi'] = max(15, features['rsi'] - 5)
        
        # Update MACD
        if 'macd' in features:
            if prediction['prediction'] == 'buy':
                features['macd'] = max(0, features['macd'] + 0.5)
            else:
                features['macd'] = min(0, features['macd'] - 0.5)
        
        # This is a simplified simulation - in a real implementation we would
        # use more sophisticated methods to predict how features would evolve
    
    def _calculate_model_trust_score(self):
        """Calculate a trust score for the model based on its performance history"""
        # Base score on recent accuracy
        if not self.learning_history['accuracy_over_time']:
            return 0.5  # Default trust if no history
            
        # Get recent accuracy values (last 5 if available)
        recent_accuracies = [entry['accuracy'] for entry in 
                            self.learning_history['accuracy_over_time'][-5:]]
        
        if not recent_accuracies:
            return 0.5
            
        # Average recent accuracy
        avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
        
        # Trust score is a combination of accuracy and learning iterations
        iterations_factor = min(1.0, self.learning_history['learning_iterations'] / 10)  # Max out at 10 iterations
        
        trust_score = (avg_accuracy * 0.7) + (iterations_factor * 0.3)
        
        return trust_score
    
    def save_model(self, filepath=None):
        """Save the trained model, scaler, and feature list"""
        if not filepath:
            filepath = f"{self.model_dir}/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        if self.model:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'indicator_performance': self.indicator_performance
                }, f)
            print(f"Model saved to {filepath}")
            return filepath
        else:
            print("No model to save")
            return None
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath and os.path.exists(filepath):
            model_path = filepath
        else:
            # Find the most recent model file
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_') and f.endswith('.pkl')]
            if not model_files:
                print("No saved model found.")
                return False
            
            # Sort by creation time (newest first)
            model_files.sort(reverse=True)
            model_path = os.path.join(self.model_dir, model_files[0])
        
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                
            self.model = saved_data.get('model')
            self.scaler = saved_data.get('scaler', StandardScaler())
            self.feature_columns = saved_data.get('feature_columns', self.feature_columns)
            self.indicator_performance = saved_data.get('indicator_performance', self.indicator_performance)
            
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
