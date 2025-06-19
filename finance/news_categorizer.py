import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# OpenAI API endpoint
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
CATEGORIES = ["Technology", "Health", "Finance", "Sports", "Entertainment", "Science", "Other"]

def fetch_news(topic, api_key, page_size=10):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={topic}&language=en&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}"
    )
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"NewsAPI error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

def is_relevant(article, topic):
    """Simple keyword-based relevance filter."""
    topic_lower = topic.lower()
    for field in ['title', 'description', 'content']:
        value = article.get(field, "")
        if value and topic_lower in value.lower():
            return True
    return False

def summarize_and_categorize(text, api_key):
    prompt = (
        f"Summarize the following news article in 2-3 sentences. "
        f"Then, categorize it as one of these: {', '.join(CATEGORIES)}. "
        f"Return your answer as:\nSummary: <summary>\nCategory: <category>\n\nArticle:\n{text}"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for news summarization and categorization."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.5
    }
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            summary, category = "No summary.", "Other"
            for line in content.splitlines():
                if line.lower().startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.lower().startswith("category:"):
                    category = line.split(":", 1)[1].strip()
            return summary, category
        else:
            return f"Failed to summarize. ({response.status_code})", "Other"
    except Exception as e:
        return f"Error: {e}", "Other"

# Streamlit UI
st.title("Real-Time News Summarizer & Categorizer (LLM-powered)")
st.write("Enter a topic to fetch the latest news, summarize, and categorize them using an LLM.")

topic = st.text_input("Enter a news topic (e.g., AI, climate, finance):", "AI")

if not NEWS_API_KEY or not OPENAI_API_KEY:
    st.warning("Please set your NEWS_API_KEY and OPENAI_API_KEY in your environment or Streamlit secrets.")
else:
    if st.button("Fetch and Summarize News"):
        with st.spinner("Fetching news articles..."):
            articles = fetch_news(topic, NEWS_API_KEY)
        if not articles:
            st.info("No articles found for this topic.")
        else:
            relevant_articles = [a for a in articles if is_relevant(a, topic)]
            if not relevant_articles:
                st.info("No relevant articles found for this topic. Showing all fetched articles instead.")
                relevant_articles = articles
            for idx, article in enumerate(relevant_articles, 1):
                title = article.get('title', 'No Title')
                description = article.get('description', '')
                content = article.get('content', '')
                url = article.get('url', '')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]
                # Use as much content as possible for summarization
                text = content or description or title
                st.markdown(f"### Article {idx}: {title}")
                st.write(f"**Source:** {source}")
                st.write(f"**Published:** {published}")
                st.write(f"**URL:** {url}")
                with st.spinner("Summarizing and categorizing..."):
                    summary, category = summarize_and_categorize(text, OPENAI_API_KEY)
                st.success(f"**Summary:** {summary}")
                st.info(f"**Category:** {category}")
                st.markdown("---")

st.caption("Powered by NewsAPI and OpenAI GPT. For demo purposes only.")
