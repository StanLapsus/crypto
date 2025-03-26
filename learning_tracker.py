import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

class LearningTracker:
    """Class to track and visualize model learning progress"""
    
    def __init__(self, model_dir='models', discord_logger=None):
        self.model_dir = model_dir
        self.discord_logger = discord_logger
        self.history_path = os.path.join(model_dir, 'learning_history.json')
        
        # Initialize history structure
        self.history = {
            'accuracy_over_time': [],
            'feature_importance_over_time': [],
            'predictions': [],
            'retrain_events': [],
            'learning_iterations': 0,
            'human_feedback': [],
            'anomalies_detected': []
        }
        
        # Load existing history if available
        self._load_history()
    
    def _load_history(self):
        """Load learning history from file"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.history = json.load(f)
                print(f"Loaded learning history: {self.history['learning_iterations']} past learning iterations")
            except Exception as e:
                print(f"Error loading learning history: {e}")
    
    def save_history(self):
        """Save learning history to file"""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving learning history: {e}")
            return False
    
    def add_training_result(self, metrics, feature_importance=None, samples_used=0, is_retraining=False):
        """Add a training event to history"""
        # Record training result
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'is_retraining': is_retraining,
            'metrics': metrics,
            'samples_used': samples_used
        }
        
        # Add feature importance if provided
        if feature_importance:
            training_record['feature_importance'] = feature_importance
        
        # Add to history
        self.history['retrain_events'].append(training_record)
        self.history['accuracy_over_time'].append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(metrics.get('accuracy', 0))
        })
        
        if feature_importance:
            # Only record top features for brevity
            top_features = feature_importance[:min(3, len(feature_importance))]
            self.history['feature_importance_over_time'].append({
                'timestamp': datetime.now().isoformat(),
                'importance': top_features
            })
        
        # Increment iterations counter
        self.history['learning_iterations'] += 1
        
        # Save updated history
        self.save_history()
        
        # Log to Discord if available
        if self.discord_logger:
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            log_message = (
                f"📊 {'Retraining' if is_retraining else 'Training'} completed\n"
                f"Accuracy: {accuracy*100:.1f}%, Precision: {precision*100:.1f}%, Recall: {recall*100:.1f}%\n"
                f"Using {samples_used} samples"
            )
            
            self.discord_logger.send_log(log_message, level="LEARNING")
            
            # Generate and send chart after a brief delay
            self.generate_learning_chart()
    
    def add_human_feedback(self, prediction_data, actual_outcome, user_info=None):
        """Record human feedback for a prediction"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_data.get('signal', 'unknown'),
            'confidence': prediction_data.get('confidence', 0),
            'actual_outcome': actual_outcome,
            'was_correct': prediction_data.get('signal', '') == actual_outcome,
            'user_info': user_info or {}
        }
        
        self.history['human_feedback'].append(feedback)
        self.save_history()
        
        # Log feedback to Discord
        if self.discord_logger:
            correct_emoji = "✅" if feedback['was_correct'] else "❌"
            log_message = (
                f"{correct_emoji} Human feedback received: {actual_outcome.upper()}\n"
                f"Original prediction: {prediction_data.get('signal', 'unknown').upper()} "
                f"({prediction_data.get('confidence', 0)*100:.1f}% confidence)"
            )
            
            self.discord_logger.send_log(log_message, level="LEARNING")
    
    def add_anomaly(self, anomaly_data):
        """Record a detected market anomaly"""
        self.history['anomalies_detected'].append(anomaly_data)
        self.save_history()
        
        # Log to Discord
        if self.discord_logger:
            log_message = (
                f"🔍 Market anomaly detected!\n"
                f"Model correctly predicted {anomaly_data.get('model_prediction', 'unknown')} "
                f"while indicators suggested {anomaly_data.get('indicator_signal', 'unknown')}"
            )
            
            self.discord_logger.send_log(log_message, level="LEARNING")
    
    def generate_learning_chart(self):
        """Generate a chart showing learning progress"""
        try:
            # Accuracy chart
            if len(self.history['accuracy_over_time']) < 2:
                return None  # Not enough data
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Extract accuracy data
            timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in self.history['accuracy_over_time']]
            accuracies = [entry['accuracy'] for entry in self.history['accuracy_over_time']]
            
            # Plot accuracy trend
            ax1.plot(timestamps, accuracies, 'o-', color='blue')
            ax1.set_title('Model Accuracy Over Time')
            ax1.set_ylabel('Accuracy')
            ax1.grid(True, alpha=0.3)
            
            # Highlight retraining events
            retrain_times = [datetime.fromisoformat(entry['timestamp']) 
                            for entry in self.history['retrain_events'] 
                            if entry.get('is_retraining', False)]
            
            for rt in retrain_times:
                ax1.axvline(x=rt, color='red', linestyle='--', alpha=0.3)
            
            # Plot feature importance if available
            if len(self.history['feature_importance_over_time']) >= 2:
                # Create a dataframe from feature importance history
                feature_data = []
                
                for entry in self.history['feature_importance_over_time']:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    for feature in entry['importance']:
                        feature_data.append({
                            'timestamp': timestamp,
                            'feature': feature.get('feature', 'unknown'),
                            'importance': feature.get('importance', 0)
                        })
                
                if feature_data:
                    feature_df = pd.DataFrame(feature_data)
                    
                    # Get top 3 features by average importance
                    top_features = feature_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(3).index
                    
                    # Plot each top feature
                    for feature in top_features:
                        feature_points = feature_df[feature_df['feature'] == feature]
                        ax2.plot(feature_points['timestamp'], feature_points['importance'], 'o-', label=feature)
                    
                    ax2.set_title('Feature Importance Over Time')
                    ax2.set_ylabel('Importance')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            
            # Send chart to Discord
            if self.discord_logger:
                self.discord_logger.send_log("Model Learning Progress Chart", level="LEARNING", chart=buf)
            
            return buf
            
        except Exception as e:
            print(f"Error generating learning chart: {e}")
            return None
    
    def calculate_trust_score(self):
        """Calculate a trust score for the model based on performance history"""
        if not self.history['accuracy_over_time']:
            return 0.5  # Default neutral trust
        
        # Calculate weighted average of recent accuracy (more weight to recent results)
        accuracies = [(datetime.fromisoformat(entry['timestamp']), entry['accuracy']) 
                      for entry in self.history['accuracy_over_time']]
        
        # Sort by timestamp
        accuracies.sort(key=lambda x: x[0])
        
        # Calculate days since each accuracy measurement
        now = datetime.now()
        weighted_accuracies = []
        
        for date, accuracy in accuracies:
            days_ago = (now - date).days
            # Exponential decay weight based on recency
            weight = np.exp(-days_ago / 30)  # 30-day half-life
            weighted_accuracies.append((weight, accuracy))
        
        # Calculate weighted average
        if weighted_accuracies:
            total_weight = sum(w for w, _ in weighted_accuracies)
            if total_weight > 0:
                weighted_avg = sum(w * a for w, a in weighted_accuracies) / total_weight
            else:
                weighted_avg = 0.5
        else:
            weighted_avg = 0.5
        
        # Factor in amount of learning
        iterations_factor = min(1.0, self.history['learning_iterations'] / 10)  # Max out at 10 iterations
        
        # Combine accuracy and learning iterations
        trust_score = (weighted_avg * 0.7) + (iterations_factor * 0.3)
        
        return trust_score
    
    def get_summary(self):
        """Get a summary of learning progress"""
        trust_score = self.calculate_trust_score()
        
        # Calculate recent accuracy trend
        if len(self.history['accuracy_over_time']) >= 2:
            recent_accuracies = [entry['accuracy'] for entry in self.history['accuracy_over_time'][-5:]]
            avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
            
            if len(recent_accuracies) >= 2:
                trend = "improving" if recent_accuracies[-1] > recent_accuracies[-2] else "declining"
            else:
                trend = "stable"
        else:
            avg_accuracy = 0
            trend = "unknown"
        
        # Human feedback stats
        total_feedback = len(self.history['human_feedback'])
        if total_feedback > 0:
            correct_feedback = sum(1 for f in self.history['human_feedback'] if f.get('was_correct', False))
            human_accuracy = correct_feedback / total_feedback
        else:
            human_accuracy = 0
        
        # Last retrain time
        if self.history['retrain_events']:
            last_retrain = datetime.fromisoformat(self.history['retrain_events'][-1]['timestamp'])
            days_since_retrain = (datetime.now() - last_retrain).days
        else:
            days_since_retrain = None
        
        # Get top features
        top_features = []
        if self.history['feature_importance_over_time']:
            latest_features = self.history['feature_importance_over_time'][-1]['importance']
            top_features = [f.get('feature', 'unknown') for f in latest_features]
        
        return {
            'trust_score': trust_score,
            'learning_iterations': self.history['learning_iterations'],
            'accuracy': avg_accuracy,
            'accuracy_trend': trend,
            'human_feedback_count': total_feedback,
            'human_feedback_accuracy': human_accuracy,
            'days_since_retrain': days_since_retrain,
            'anomalies_detected': len(self.history['anomalies_detected']),
            'top_features': top_features
        }
