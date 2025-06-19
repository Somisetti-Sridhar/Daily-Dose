import os
import streamlit as st
import requests

# Optional: Use dotenv for local development
from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
CATEGORIES = ["Technology", "Health", "Finance", "Sports", "Entertainment", "Science", "Other"]

def fetch_news(topic, api_key, page_size=10):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={topic}&language=en&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        st.error(f"Failed to fetch news articles. Status code: {response.status_code}")
        return []

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
            return "Failed to summarize.", "Other"
    except Exception as e:
        return f"Error: {e}", "Other"

st.title("Real-Time News Summarizer & Categorizer (LLM-powered)")
st.write("Enter a topic to fetch the latest news, summarize, and categorize them using an LLM.")

topic = st.text_input("Enter a news topic (e.g., AI, climate, finance):", "AI")

if not NEWS_API_KEY or not OPENAI_API_KEY:
    st.warning("Please set your NEWS_API_KEY and OPENAI_API_KEY in your environment or Streamlit secrets.")
else:
    if st.button("Fetch and Summarize News"):
        with st.spinner("Fetching news articles..."):
            articles = fetch_news(topic, NEWS_API_KEY)
        if articles:
            found = False
            for idx, article in enumerate(articles, 1):
                # Use as much content as possible for summarization
                text = article.get('content') or article.get('description') or article.get('title')
                if not text or text.strip() == "":
                    continue
                found = True
                st.markdown(f"### Article {idx}: {article.get('title', 'No Title')}")
                st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                st.write(f"**Published:** {article.get('publishedAt', '')[:10]}")
                st.write(f"**URL:** {article.get('url', '')}")
                with st.spinner("Summarizing and categorizing..."):
                    summary, category = summarize_and_categorize(text, OPENAI_API_KEY)
                st.success(f"**Summary:** {summary}")
                st.info(f"**Category:** {category}")
                st.markdown("---")
            if not found:
                st.info("No articles with enough content to summarize were found for this topic.")
        else:
            st.info("No articles found for this topic.")

st.caption("Powered by NewsAPI and OpenAI GPT. For demo purposes only.")
