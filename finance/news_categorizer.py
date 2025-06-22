import streamlit as st
import feedparser
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def fetch_google_news_rss(query, max_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.title,
            "description": entry.get("summary", ""),
            "url": entry.link,
            "published": entry.get("published", "")
        })
    return articles

def summarize_text(text):
    if not text or len(text.split()) < 40:
        return text  # Too short to summarize, return as-is
    try:
        summary = summarizer(text, max_length=130, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {e}"

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

st.title("📰 Google News Summarizer & Categorizer (BART Transformer)")
st.write("Fetches the latest headlines from Google News RSS and summarizes them with a state-of-the-art transformer model (BART).")

topic = st.text_input("Enter a news topic (e.g., AI, climate, finance):", "AI")
num_articles = st.slider("Number of articles to show", 3, 20, 5)

if st.button("Fetch and Summarize News"):
    with st.spinner("Fetching news articles..."):
        articles = fetch_google_news_rss(topic, num_articles)
    if not articles:
        st.error("No news articles found for this topic. Try a broader or trending topic.")
    else:
        for idx, article in enumerate(articles, 1):
            title = article.get('title', 'No Title')
            desc = article.get('description', '')
            url = article.get('url', '')
            published = article.get('published', '')
            st.markdown(f"### Article {idx}: {title}")
            st.write(f"**Published:** {published}")
            st.write(f"**URL:** {url}")
            with st.spinner("Summarizing..."):
                summary = summarize_text(desc)
            category = categorize(summary)
            st.success(f"**Summary:** {summary}")
            st.info(f"**Category:** {category}")
            st.markdown("---")

st.caption("Powered by Google News RSS and Hugging Face Transformers (facebook/bart-large-cnn).")
