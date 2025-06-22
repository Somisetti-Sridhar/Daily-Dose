import os
import streamlit as st
import requests
from transformers import pipeline

NEWS_API_KEY = os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY", "")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def fetch_news(topic, api_key, page_size=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={topic}&language=en&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if resp.status_code == 200 and data.get("status") == "ok":
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "content": a.get("content", ""),
                    "url": a.get("url", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "published": a.get("publishedAt", "")[:10]
                }
                for a in data.get("articles", []) if a.get("title")
            ]
        else:
            st.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"NewsAPI error: {e}")
        return []

def summarize_text(text, max_length=130):
    if not text or len(text.split()) < 40:
        return text
    summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def categorize(text):
    categories = {
        "Technology": ["AI", "technology", "software", "robot", "computer", "internet"],
        "Health": ["health", "medicine", "covid", "hospital", "doctor", "disease"],
        "Finance": ["finance", "stock", "market", "investment", "bank", "cryptocurrency"],
        "Sports": ["sport", "football", "cricket", "tennis", "olympic"],
        "Entertainment": ["movie", "music", "entertainment", "celebrity", "tv", "film"],
        "Science": ["science", "research", "space", "nasa", "physics", "biology"]
    }
    text_lower = text.lower()
    for cat, keywords in categories.items():
        if any(word in text_lower for word in keywords):
            return cat
    return "Other"

st.title("📰 News Summarizer & Categorizer (BART Transformer)")
st.write("Fetches the latest headlines using NewsAPI and summarizes them with a state-of-the-art transformer model (BART).")

topic = st.text_input("Enter a news topic (e.g., AI, climate, finance):", "AI")
num_articles = st.slider("Number of articles to show", 3, 15, 5)

if not NEWS_API_KEY:
    st.warning("Please set your NEWS_API_KEY in your environment or Streamlit secrets.")
else:
    if st.button("Fetch and Summarize News"):
        with st.spinner("Fetching news articles..."):
            articles = fetch_news(topic, NEWS_API_KEY, num_articles)
        if not articles:
            st.error("No news articles found for this topic. Try a broader or trending topic.")
        else:
            for idx, article in enumerate(articles, 1):
                title = article.get('title', 'No Title')
                description = article.get('description', '')
                content = article.get('content', '')
                url = article.get('url', '')
                source = article.get('source', 'Unknown')
                published = article.get('published', '')
                text = content or description or title
                st.markdown(f"### Article {idx}: {title}")
                st.write(f"**Source:** {source}")
                st.write(f"**Published:** {published}")
                st.write(f"**URL:** {url}")
                with st.spinner("Summarizing..."):
                    summary = summarize_text(text)
                category = categorize(summary)
                st.success(f"**Summary:** {summary}")
                st.info(f"**Category:** {category}")
                st.markdown("---")

st.caption("Powered by NewsAPI and Hugging Face Transformers (facebook/bart-large-cnn).")
