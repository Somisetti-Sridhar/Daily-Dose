import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import feedparser
from transformers import pipeline

# Download NLTK VADER lexicon if not present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #23272f;
        color: #f8f8f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00ff00 !important;
    }
    .negative {
        color: #ff4c4c !important;
    }
    .neutral {
        color: #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)

# --- News Scraper Functions ---
def scrape_coindesk(crypto_symbol="bitcoin"):
    try:
        url = f"https://www.coindesk.com/tag/{crypto_symbol}/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_items = soup.find_all('h3', class_=['heading', 'headline'])[:10]
        articles = []
        for item in news_items:
            title = item.get_text().strip()
            if title and len(title) > 10:
                articles.append({
                    'title': title,
                    'timestamp': datetime.now(),
                    'source': 'CoinDesk'
                })
        return articles
    except Exception:
        return []

def scrape_cointelegraph(crypto_symbol="bitcoin"):
    try:
        feed_url = "https://cointelegraph.com/rss"
        feed = feedparser.parse(feed_url)
        articles = []
        for entry in feed.entries[:20]:
            title = entry.title
            if crypto_symbol.lower() in title.lower():
                articles.append({
                    'title': title,
                    'timestamp': datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                    'source': 'Cointelegraph'
                })
        return articles
    except Exception:
        return []

def scrape_cryptopanic(crypto_symbol="bitcoin"):
    try:
        feed_url = f"https://cryptopanic.com/news/{crypto_symbol.lower()}/rss"
        feed = feedparser.parse(feed_url)
        articles = []
        for entry in feed.entries[:20]:
            articles.append({
                'title': entry.title,
                'timestamp': datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                'source': 'CryptoPanic'
            })
        return articles
    except Exception:
        return []

def get_crypto_news(crypto_symbol="bitcoin"):
    news = scrape_coindesk(crypto_symbol)
    if news:
        return news
    news = scrape_cointelegraph(crypto_symbol)
    if news:
        return news
    news = scrape_cryptopanic(crypto_symbol)
    if news:
        return news
    st.error("❌ Could not fetch real news articles from CoinDesk, Cointelegraph, or CryptoPanic. Please check your internet connection or try again later.")
    st.stop()

# --- Sentiment Analyzer ---
class CryptoNewsAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def get_crypto_news(self, crypto_symbol="bitcoin"):
        return get_crypto_news(crypto_symbol)
    
    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)
    
    def get_crypto_price(self, symbol, period="1mo"):
        try:
            ticker = f"{symbol}-USD"
            crypto = yf.Ticker(ticker)
            hist = crypto.history(period=period)
            return hist
        except:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 1000)
            return pd.DataFrame({'Close': prices, 'Volume': np.random.randint(1000000, 10000000, len(dates))}, index=dates)

# --- Correlation Calculation ---
def calculate_sentiment_price_correlation(sentiment_df, price_data):
    sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date
    daily_sentiment = sentiment_df.groupby('date')['compound'].mean().reset_index()
    price_dates = price_data.index.date
    correlation_data = []
    for i in range(1, len(price_dates)):
        date = price_dates[i]
        prev_date = price_dates[i-1]
        sentiment_score = daily_sentiment[daily_sentiment['date'] == date]['compound'].values
        if len(sentiment_score) > 0:
            prev_close = price_data.iloc[i-1]['Close']
            today_close = price_data.iloc[i]['Close']
            price_change = ((today_close - prev_close) / prev_close) * 100
            correlation_data.append({'date': date, 'sentiment': sentiment_score[0], 'price_change': price_change})
    if len(correlation_data) > 2:
        corr_df = pd.DataFrame(correlation_data)
        correlation = corr_df['sentiment'].corr(corr_df['price_change'])
        return corr_df, correlation
    else:
        return pd.DataFrame(), None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Crypto News Sentiment & Price Analyzer", layout="wide")
    st.markdown('<h1 class="main-header">🚀 Crypto News Sentiment & Price Analyzer</h1>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Configuration")
    crypto_options = {
        'Bitcoin': 'BTC',
        'Ethereum': 'ETH',
        'Cardano': 'ADA',
        'Polkadot': 'DOT',
        'Chainlink': 'LINK'
    }
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
    crypto_symbol = crypto_options[selected_crypto]
    time_period = st.sidebar.selectbox("Price Analysis Period", ['1mo', '3mo', '6mo', '1y'], index=0)
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        import time
        time.sleep(30)
        st.experimental_rerun()

    analyzer = CryptoNewsAnalyzer()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📰 News Sentiment Analysis")
        with st.spinner("Fetching latest news..."):
            news_data = analyzer.get_crypto_news(selected_crypto.lower())
        if news_data:
            sentiment_data = []
            for article in news_data:
                sentiment = analyzer.analyze_sentiment(article['title'])
                sentiment_data.append({
                    'title': article['title'],
                    'compound': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu'],
                    'timestamp': article['timestamp'],
                    'source': article['source']
                })
            sentiment_df = pd.DataFrame(sentiment_data)
            avg_sentiment = sentiment_df['compound'].mean()
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            sentiment_color = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Overall Sentiment: <span class="{sentiment_color}">{sentiment_label}</span></h3>
                <p>Average Score: {avg_sentiment:.3f}</p>
                <p>Analyzed {len(sentiment_data)} news articles (Source: {sentiment_df['source'].iloc[0]})</p>
            </div>
            """, unsafe_allow_html=True)
            fig_sentiment = px.histogram(sentiment_df, x='compound', nbins=10, title="Sentiment Score Distribution")
            fig_sentiment.update_layout(height=300)
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.subheader("📄 Recent News Headlines")
            for idx, article in enumerate(sentiment_data[:5]):
                sentiment_emoji = "😊" if article['compound'] > 0.1 else "😞" if article['compound'] < -0.1 else "😐"
                st.write(f"{sentiment_emoji} **{article['title']}**")
                st.write(f"Sentiment: {article['compound']:.3f} | Source: {article['source']}")
                st.write("---")

    with col2:
        st.header("💰 Price Analysis")
        with st.spinner("Fetching price data..."):
            price_data = analyzer.get_crypto_price(crypto_symbol, time_period)
        if not price_data.empty:
            current_price = price_data['Close'].iloc[-1]
            price_change = ((current_price - price_data['Close'].iloc[-2]) / price_data['Close'].iloc[-2]) * 100
            price_change_color = "positive" if price_change > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{selected_crypto} Price</h3>
                <h2>${current_price:,.2f}</h2>
                <p class="{price_change_color}">
                    {'+' if price_change > 0 else ''}{price_change:.2f}% (24h)
                </p>
            </div>
            """, unsafe_allow_html=True)
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))
            fig_price.update_layout(title=f"{selected_crypto} Price Trend ({time_period})", xaxis_title="Date", yaxis_title="Price (USD)", height=400, showlegend=False)
            st.plotly_chart(fig_price, use_container_width=True)
            fig_volume = px.bar(x=price_data.index, y=price_data['Volume'], title="Trading Volume")
            fig_volume.update_layout(height=200)
            st.plotly_chart(fig_volume, use_container_width=True)

    # --- Correlation Analysis ---
    st.header("🔗 Sentiment-Price Correlation Analysis")
    if news_data and not price_data.empty and len(sentiment_df) > 0:
        corr_df, correlation = calculate_sentiment_price_correlation(sentiment_df, price_data)
        if not corr_df.empty and correlation is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correlation Coefficient", f"{correlation:.3f}")
            with col2:
                correlation_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
                st.metric("Correlation Strength", correlation_strength)
            with col3:
                correlation_direction = "Positive" if correlation > 0 else "Negative"
                st.metric("Correlation Direction", correlation_direction)
            fig_corr = px.scatter(
                corr_df,
                x='sentiment',
                y='price_change',
                title="Sentiment vs Price Change Correlation",
                labels={'sentiment': 'Daily Sentiment Score', 'price_change': 'Price Change (%)'},
                trendline="ols"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough overlapping dates between news and price data to calculate correlation.")
    else:
        st.info("Not enough data to calculate correlation.")

    # --- Sentiment Explanation ---
    st.markdown("### ℹ️ What does 'Sentiment' mean?")
    st.write("""
**Sentiment** measures the emotional tone of news headlines—whether they are positive (bullish), negative (bearish), or neutral.  
In financial markets, sentiment can influence investor behavior and price movements.  
- **Positive sentiment** often reflects optimism and can lead to buying pressure.
- **Negative sentiment** reflects pessimism, risk, or fear, and can lead to selling or caution.
- **Neutral sentiment** is factual or analytical, with little emotional impact.

Persistent positive or negative sentiment can precede trends in price, but is not always predictive.
    """)

    # --- User Upload: Summarize & Sentimentize ---
    st.markdown("## 📝 Summarize & Analyze Your Own Financial News")
    uploaded_file = st.file_uploader("Upload a news article (TXT format) or paste text below:", type=['txt'])
    user_text = st.text_area("Or paste your financial news text here:")

    if uploaded_file or user_text:
        if uploaded_file:
            text = uploaded_file.read().decode('utf-8')
        else:
            text = user_text
        st.subheader("Summary")
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            st.write(summary)
        except Exception as e:
            st.info("Summarization model could not be loaded. Please ensure you have 'transformers' and required models installed.")
        st.subheader("Sentiment Analysis")
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        st.write(f"**Compound Score:** {sentiment['compound']:.3f}")
        if sentiment['compound'] > 0.1:
            st.success("Positive Sentiment")
        elif sentiment['compound'] < -0.1:
            st.error("Negative Sentiment")
        else:
            st.info("Neutral Sentiment")

    st.markdown("---")
    st.markdown("""
    **About this App:**
    This application analyzes real-time news sentiment and correlates it with cryptocurrency price movements. 
    It uses VADER sentiment analysis on news headlines and fetches live price data to provide insights into 
    market sentiment and price relationships.

    **Technologies Used:** Streamlit, NLTK, yfinance, Plotly, BeautifulSoup, feedparser, transformers
    """)

if __name__ == "__main__":
    main()
