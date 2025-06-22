import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser

NEWS_API_KEY = os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY", "")
NEWSDATA_KEY = os.getenv("NEWSDATA_KEY") or st.secrets.get("NEWSDATA_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
CATEGORIES = ["Technology", "Health", "Finance", "Sports", "Entertainment", "Science", "Other"]

def fetch_newsapi(query, api_key, max_articles=10):
    url = (
        f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={max_articles}&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if resp.status_code == 429 or data.get("code") in ["rateLimited", "apiKeyExhausted"]:
            st.warning("NewsAPI rate limit reached or key exhausted. Try another key or wait.")
            return []
        if data.get("status") != "ok":
            st.warning(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "content": a.get("content", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "published": a.get("publishedAt", "")[:10]
            }
            for a in data.get("articles", []) if "title" in a
        ]
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []

def fetch_newsdata(query, api_key, max_articles=10):
    url = (
        f"https://newsdata.io/api/1/news?apikey={api_key}&q={query}&language=en"
        f"&country=us,gb,ca,au&category=top,technology,science"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "content": a.get("content", ""),
                "url": a.get("link", ""),
                "source": a.get("source_id", "NewsData.io"),
                "published": a.get("pubDate", "")[:10]
            }
            for a in data.get("results", []) if "title" in a
        ][:max_articles]
    except Exception as e:
        st.warning(f"NewsData.io error: {e}")
        return []

def scrape_bbc(query, max_articles=10):
    url = f"https://www.bbc.co.uk/search?q={query}&filter=news"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for h in soup.select("article h1, article h2, article h3"):
            title = h.text.strip()
            link = h.find_parent("a")["href"] if h.find_parent("a") else ""
            articles.append({
                "title": title,
                "description": "",
                "content": "",
                "url": link,
                "source": "BBC",
                "published": ""
            })
            if len(articles) >= max_articles:
                break
        return articles
    except Exception as e:
        st.warning(f"BBC scrape error: {e}")
        return []

def scrape_reuters(query, max_articles=10):
    url = f"https://www.reuters.com/site-search/?query={query}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for h in soup.select("h3.search-result-title"):
            title = h.text.strip()
            link = h.find_parent("a")["href"] if h.find_parent("a") else ""
            articles.append({
                "title": title,
                "description": "",
                "content": "",
                "url": f"https://www.reuters.com{link}" if link and link.startswith("/") else link,
                "source": "Reuters",
                "published": ""
            })
            if len(articles) >= max_articles:
                break
        return articles
    except Exception as e:
        st.warning(f"Reuters scrape error: {e}")
        return []

def fetch_google_news_rss(query, max_articles=10):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.title,
                "description": entry.get("summary", ""),
                "content": "",
                "url": entry.link,
                "source": "Google News",
                "published": entry.get("published", "")[:10]
            })
        return articles
    except Exception as e:
        st.warning(f"Google News RSS error: {e}")
        return []

def get_headlines(query, newsapi_key, newsdata_key, max_articles=20):
    sources = []
    sources += fetch_newsapi(query, newsapi_key, max_articles // 2)
    sources += fetch_newsdata(query, newsdata_key, max_articles // 2)
    sources += scrape_bbc(query, max_articles // 4)
    sources += scrape_reuters(query, max_articles // 4)
    sources += fetch_google_news_rss(query, max_articles // 4)
    seen = set()
    unique = []
    for s in sources:
        key = (s["title"], s["url"])
        if s["title"] and key not in seen:
            unique.append(s)
            seen.add(key)
    return unique[:max_articles]

import time

def summarize_and_categorize(text, api_key, max_retries=3):
    prompt = (
        f"Summarize the following news article in 2-3 sentences. "
        f"Then, categorize it as one of these: Technology, Health, Finance, Sports, Entertainment, Science, Other. "
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
    for attempt in range(max_retries):
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
            elif response.status_code == 429:
                # Too many requests, wait and retry
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return f"Failed to summarize. ({response.status_code})", "Other"
        except Exception as e:
            return f"Error: {e}", "Other"
    return "Failed to summarize due to rate limiting (429). Try again later.", "Other"


# ---- Streamlit UI ----
st.title("📰 Multi-Source News Summarizer & Categorizer (LLM-powered)")
st.write("Fetches the latest headlines from NewsAPI, NewsData.io, BBC, Reuters, and Google News. Summarizes and categorizes each article using OpenAI GPT.")

topic = st.text_input("Enter a news topic (e.g., AI, climate, finance):", "AI")
num_articles = st.slider("Number of articles to show", 5, 30, 15)

if not NEWS_API_KEY or not NEWSDATA_KEY or not OPENAI_API_KEY:
    st.warning("Please set your NEWS_API_KEY, NEWSDATA_KEY, and OPENAI_API_KEY in your environment or Streamlit secrets.")
else:
    if st.button("Fetch and Summarize News"):
        with st.spinner("Fetching news articles from multiple sources..."):
            articles = get_headlines(topic, NEWS_API_KEY, NEWSDATA_KEY, num_articles)
        if not articles:
            st.error("No news articles found for this topic across all sources. Try a broader or trending topic.")
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
                with st.spinner("Summarizing and categorizing..."):
                    summary, category = summarize_and_categorize(text, OPENAI_API_KEY)
                st.success(f"**Summary:** {summary}")
                st.info(f"**Category:** {category}")
                st.markdown("---")

st.caption("Powered by NewsAPI, NewsData.io, BBC, Reuters, Google News, and OpenAI GPT.")
