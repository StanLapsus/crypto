import requests
import os
import json
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from api_utils import CryptoAPIUtils

class NewsSentimentAnalyzer:
    def __init__(self, news_api_key="your_default_newsapi_key_here"):
        self.news_api_key = news_api_key or os.environ.get('NEWS_API_KEY')
        self.news_cache = {}
        self.sentiment_analyzer = None
        self.fallback_sources = ['cryptopanic', 'twitter_trends', 'reddit']
        
        # Initialize NLTK components if needed
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Add API utils for free API access
        self.api_utils = CryptoAPIUtils()
    
    def fetch_crypto_news(self, query="bitcoin OR cryptocurrency", days=1):
        """Fetch crypto-related news from NewsAPI with fallbacks if unavailable"""
        # Try NewsAPI first if key is available
        if self.news_api_key:
            news = self._fetch_from_newsapi(query, days)
            if news:
                return news
                
        # If NewsAPI failed or no key, try alternative sources
        print("Primary news source unavailable, trying alternatives...")
        
        # Try CryptoPanic API (free for basic usage)
        cryptopanic_news = self._fetch_from_cryptopanic(query, days)
        if cryptopanic_news:
            return cryptopanic_news
            
        # Try Reddit scraping as another alternative
        reddit_news = self._fetch_from_reddit(query, days)
        if reddit_news:
            return reddit_news
            
        # No news sources available
        print("All news sources failed. Proceeding without news data.")
        return []
    
    def _fetch_from_newsapi(self, query, days):
        """Fetch from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'ok':
                return data.get('articles', [])
            else:
                print(f"Error fetching news from NewsAPI: {data.get('message')}")
                return []
        except Exception as e:
            print(f"Error fetching news from NewsAPI: {e}")
            return []
    
    def _fetch_from_cryptopanic(self, query, days):
        """Fetch from CryptoPanic API (free public feed)"""
        try:
            # CryptoPanic has a public API that doesn't require authentication for basic feeds
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': '',  # Leave empty for public feed
                'currencies': 'BTC' if 'bitcoin' in query.lower() else '',
                'kind': 'news',
                'public': 'true'
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"CryptoPanic API error: {response.status_code}")
                return []
                
            data = response.json()
            
            # Convert to format similar to NewsAPI
            articles = []
            for result in data.get('results', []):
                articles.append({
                    'title': result.get('title', ''),
                    'description': result.get('title', ''),  # No separate description in CryptoPanic
                    'url': result.get('url', ''),
                    'publishedAt': result.get('created_at', ''),
                    'source': {'name': result.get('source', {}).get('title', 'CryptoPanic')}
                })
                
            return articles
        except Exception as e:
            print(f"Error fetching news from CryptoPanic: {e}")
            return []
    
    def _fetch_from_reddit(self, query, days):
        """Fetch crypto news from Reddit (without authentication)"""
        try:
            # Reddit's public JSON API
            subreddits = ['CryptoCurrency', 'Bitcoin', 'CryptoMarkets']
            
            articles = []
            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    continue
                    
                data = response.json()
                
                for post in data.get('data', {}).get('children', [])[:10]:  # Get top 10 posts
                    post_data = post.get('data', {})
                    
                    # Skip non-text posts
                    if post_data.get('is_video', False) or not post_data.get('selftext'):
                        continue
                    
                    # Convert to our standard format
                    articles.append({
                        'title': post_data.get('title', ''),
                        'description': post_data.get('selftext', '')[:300] + '...',
                        'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
                        'publishedAt': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                        'source': {'name': f"Reddit r/{subreddit}"}
                    })
            
            return articles
        except Exception as e:
            print(f"Error fetching news from Reddit: {e}")
            return []
    
    def analyze_sentiment_with_textblob(self, text):
        """Analyze news sentiment using TextBlob"""
        try:
            # Basic sentiment analysis with TextBlob
            blob = TextBlob(text)
            # TextBlob sentiment: polarity between -1.0 and 1.0
            polarity = blob.sentiment.polarity
            
            # Map polarity to our -1 to 1 score
            if polarity > 0.2:
                sentiment = "bullish"
            elif polarity < -0.2:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            # Extract relevant crypto terms for impact analysis
            crypto_terms = ['bitcoin', 'crypto', 'blockchain', 'eth', 'btc', 'altcoin', 
                          'mining', 'wallet', 'exchange', 'defi', 'token']
            
            # Calculate impact based on sentence structure and crypto terms
            impact = polarity
            if any(term in text.lower() for term in crypto_terms):
                impact *= 1.5  # Amplify if crypto-specific
            
            result = {
                "sentiment": sentiment,
                "score": polarity,
                "market_impact": self._generate_impact_text(polarity, text)
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing sentiment with TextBlob: {e}")
            return {"sentiment": "neutral", "score": 0.0, "market_impact": "Error in analysis"}
    
    def analyze_sentiment_with_vader(self, text):
        """Analyze news sentiment using NLTK's VADER"""
        try:
            # Get sentiment scores
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Extract the compound score (-1 to 1)
            compound = scores['compound']
            
            # Map the compound score to our sentiment categories
            if compound > 0.2:
                sentiment = "bullish"
            elif compound < -0.2:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            result = {
                "sentiment": sentiment,
                "score": compound,
                "market_impact": self._generate_impact_text(compound, text)
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing sentiment with VADER: {e}")
            return {"sentiment": "neutral", "score": 0.0, "market_impact": "Error in analysis"}
    
    def _generate_impact_text(self, score, text):
        """Generate market impact text based on sentiment score"""
        # Extract key sentences for summary
        sentences = TextBlob(text).sentences
        
        if not sentences:
            return "No clear market impact detected."
        
        # Use the most polar sentence for the impact description
        key_sentence = max(sentences, key=lambda s: abs(s.sentiment.polarity))
        key_sentence_text = str(key_sentence)
        
        # Generate impact statement based on score
        if score > 0.5:
            impact = f"Strongly bullish. {key_sentence_text}"
        elif score > 0.2:
            impact = f"Moderately bullish. {key_sentence_text}"
        elif score < -0.5:
            impact = f"Strongly bearish. {key_sentence_text}"
        elif score < -0.2:
            impact = f"Moderately bearish. {key_sentence_text}"
        else:
            impact = f"Neutral market impact. {key_sentence_text}"
        
        return impact[:200]  # Limit length
    
    def get_news_sentiment(self, query="bitcoin", days=1, max_articles=5):
        """Get aggregated sentiment from recent news with fallback options"""
        # Check cache first
        cache_key = f"{query}_{days}_{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key in self.news_cache:
            return self.news_cache[cache_key]
            
        articles = self.fetch_crypto_news(query, days)
        
        if not articles:
            # No news articles found - return neutral sentiment with warning
            no_news_result = {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "articles_analyzed": 0,
                "summary": "No news articles available. Analysis based on technical indicators only.",
                "data_available": False
            }
            
            # Cache this result
            self.news_cache[cache_key] = no_news_result
            return no_news_result
        
        # Analyze top articles
        analyzed_articles = []
        sentiment_scores = []
        
        for article in articles[:max_articles]:
            title = article.get('title', '')
            content = article.get('description', '')
            
            # Skip articles with no content
            if not title and not content:
                continue
                
            # Analyze combined title and content
            text_to_analyze = f"{title}. {content}"
            
            # Use both sentiment analyzers and average the results
            textblob_result = self.analyze_sentiment_with_textblob(text_to_analyze)
            vader_result = self.analyze_sentiment_with_vader(text_to_analyze)
            
            # Average the scores from both methods
            avg_score = (textblob_result.get('score', 0) + vader_result.get('score', 0)) / 2
            
            # Determine sentiment based on average score
            if avg_score > 0.2:
                sentiment = "bullish"
            elif avg_score < -0.2:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            sentiment_scores.append(avg_score)
            
            analyzed_articles.append({
                "title": title,
                "url": article.get('url', ''),
                "published_at": article.get('publishedAt', ''),
                "sentiment": sentiment,
                "sentiment_score": avg_score,
                "market_impact": textblob_result.get('market_impact', '')
            })
        
        # Calculate overall sentiment
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            
            if avg_score > 0.3:
                overall_sentiment = "bullish"
            elif avg_score < -0.3:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
        else:
            avg_score = 0
            overall_sentiment = "neutral"
        
        result = {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": avg_score,
            "articles_analyzed": len(analyzed_articles),
            "articles": analyzed_articles,
            "summary": f"Overall {overall_sentiment} sentiment with score {avg_score:.2f} based on {len(analyzed_articles)} articles",
            "data_available": True
        }
        
        # Cache the result
        self.news_cache[cache_key] = result
        
        return result
    
    def get_news_articles(self, query, days=1, max_articles=10):
        """Get crypto news articles without requiring an API key"""
        try:
            # Use our API utils to get free news
            news_data = self.api_utils.get_crypto_news(
                keywords=query,
                max_items=max_articles
            )
            
            if not news_data or not news_data.get('articles'):
                return []
            
            # Return the articles
            return news_data.get('articles', [])
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
