import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import spacy

import subprocess

st.set_page_config(page_title="Multi-Source News Sentiment & Graph Analytics", layout="wide")
st.title("📰 Multi-Source News Sentiment & Graph Analytics Dashboard")

# ---- CONFIGURATION ----
DEFAULT_NEWSAPI_KEY = "244801526e3845d48fbac912de6371b8"
NEWSDATA_KEY = "pub_4d611d9c53524c48a131fb526bab9220"

@st.cache_resource
def download_en_core_web_sm():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Automatically download model if not present
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

sentiment_model = load_sentiment_model()
nlp = load_spacy_model()

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
        return [a["title"] for a in data.get("articles", []) if "title" in a]
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []

def fetch_newsdata(query, max_articles=10):
    url = (
        f"https://newsdata.io/api/1/news?apikey={NEWSDATA_KEY}&q={query}&language=en"
        f"&country=us,gb,ca,au&category=top,technology,science"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return [a["title"] for a in data.get("results", []) if "title" in a][:max_articles]
    except Exception as e:
        st.warning(f"NewsData.io error: {e}")
        return []

def scrape_bbc(query, max_articles=10):
    url = f"https://www.bbc.co.uk/search?q={query}&filter=news"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [
            h.text.strip()
            for h in soup.select("article h1, article h2, article h3")
        ]
        return headlines[:max_articles]
    except Exception as e:
        st.warning(f"BBC scrape error: {e}")
        return []

def scrape_reuters(query, max_articles=10):
    url = f"https://www.reuters.com/site-search/?query={query}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [
            h.text.strip()
            for h in soup.select("h3.search-result-title")
        ]
        return headlines[:max_articles]
    except Exception as e:
        st.warning(f"Reuters scrape error: {e}")
        return []

def get_headlines(query, newsapi_key, max_articles=20):
    sources = []
    sources += fetch_newsapi(query, newsapi_key, max_articles // 2)
    sources += fetch_newsdata(query, max_articles // 2)
    sources += scrape_bbc(query, max_articles // 4)
    sources += scrape_reuters(query, max_articles // 4)
    seen = set()
    unique = []
    for s in sources:
        if s and s not in seen:
            unique.append(s)
            seen.add(s)
    return unique[:max_articles]

def analyze_sentiments(headlines):
    results = sentiment_model(headlines)
    sentiments = [r['label'] for r in results]
    scores = [r['score'] for r in results]
    return sentiments, scores

def plot_sentiment_distribution(sentiments, topic=None):
    df = pd.DataFrame(sentiments, columns=["Sentiment"])
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={"POSITIVE": "green", "NEGATIVE": "red"},
        title=f"Sentiment Distribution{f' for {topic}' if topic else ''}",
        height=350
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_sentiment_scores(scores, sentiments, topic=None):
    df = pd.DataFrame({"Score": scores, "Sentiment": sentiments})
    fig = px.histogram(
        df,
        x="Score",
        color="Sentiment",
        nbins=20,
        barmode="overlay",
        color_discrete_map={"POSITIVE": "green", "NEGATIVE": "red"},
        title=f"Sentiment Confidence Scores{f' for {topic}' if topic else ''}",
        height=350
    )
    return fig

def plot_topic_analytics(topic_sentiments):
    df = pd.DataFrame([
        {"Topic": topic, "Positive": sentiments.count("POSITIVE"), "Negative": sentiments.count("NEGATIVE")}
        for topic, sentiments in topic_sentiments.items()
    ])
    df = df.melt(id_vars="Topic", value_vars=["Positive", "Negative"], var_name="Sentiment", value_name="Count")
    fig = px.bar(
        df,
        x="Topic",
        y="Count",
        color="Sentiment",
        barmode="group",
        color_discrete_map={"Positive": "green", "Negative": "red"},
        title="Sentiment Comparison Across Topics",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def extract_entities(headlines):
    entities_per_headline = []
    for headline in headlines:
        doc = nlp(headline)
        entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
        entities_per_headline.append(entities)
    return entities_per_headline

def build_entity_graph(entities_per_headline):
    G = nx.Graph()
    for entities in entities_per_headline:
        entities = list(set(entities))  # Remove duplicates in same headline
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if G.has_edge(entity1, entity2):
                    G[entity1][entity2]["weight"] += 1
                else:
                    G.add_edge(entity1, entity2, weight=1)
    return G

def plot_networkx_graph(G, title="Entity Co-occurrence Network"):
    if len(G) == 0:
        st.info("No entities detected for network graph.")
        return
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    node_x = []
    node_y = []
    node_text = []
    node_degree = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_degree.append(G.degree(node))
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_degree,
            size=[8 + 2*d for d in node_degree],
            colorbar=dict(
                thickness=15,
                title=dict(text='Node Degree'),
                xanchor='left'
            ),
            line_width=2
        )
    )
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Entity nodes sized by degree. Edges weighted by co-occurrence.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
    st.plotly_chart(fig, use_container_width=True)

def show_graph_analytics(G):
    if len(G) == 0:
        return
    st.write("#### Graph Analytics")
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    st.table(pd.DataFrame(top_nodes, columns=["Entity", "Degree Centrality"]))
    st.write(f"Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    user_api_key = st.text_input(
        "Enter your NewsAPI key (leave blank to use default):",
        value="",
        type="password",
        help="Get a free key at https://newsapi.org if you hit rate limits."
    )
with col2:
    num_headlines = st.number_input(
        "How many headlines per topic?",
        min_value=4,
        max_value=40,
        value=20,
        step=2,
        help="Total headlines per topic, aggregated from all sources."
    )

topics = st.text_input(
    "Enter topics (comma-separated for analytics, e.g., 'AI, Climate Change, Elections'):",
    value="AI"
)

refresh = st.button("Fetch & Analyze Latest Headlines")

if refresh and topics:
    topics_list = [t.strip() for t in topics.split(",") if t.strip()]
    topic_sentiments = {}
    all_headlines = []
    all_entities_per_headline = []
    # --- Fetch all headlines first ---
    for topic in topics_list:
        headlines = get_headlines(topic, user_api_key or DEFAULT_NEWSAPI_KEY, int(num_headlines))
        all_headlines.extend(headlines)
        all_entities_per_headline.extend(extract_entities(headlines))
    if len(topics_list) > 1:
        st.markdown("## Global Entity Co-occurrence Network Across All Topics")
        G_global = build_entity_graph(all_entities_per_headline)
        plot_networkx_graph(G_global, title="Global Entity Co-occurrence Network (All Topics)")
        show_graph_analytics(G_global)
    # --- Per-topic analytics ---
    for idx, topic in enumerate(topics_list):
        st.subheader(f"Results for topic: {topic}")
        headlines = all_headlines[idx*int(num_headlines):(idx+1)*int(num_headlines)]
        if not headlines:
            st.warning(f"No headlines found for '{topic}'. Try a different topic or check your API key.")
            continue
        sentiments, scores = analyze_sentiments(headlines)
        topic_sentiments[topic] = sentiments
        st.dataframe(pd.DataFrame({
            "Headline": headlines,
            "Sentiment": sentiments,
            "Confidence": scores
        }))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_sentiment_distribution(sentiments, topic), use_container_width=True)
        with c2:
            st.plotly_chart(plot_sentiment_scores(scores, sentiments, topic), use_container_width=True)
        # --- Graph Analytics ---
        st.markdown("#### Entity Co-occurrence Network")
        entities_per_headline = all_entities_per_headline[idx*int(num_headlines):(idx+1)*int(num_headlines)]
        G = build_entity_graph(entities_per_headline)
        plot_networkx_graph(G, title=f"Entity Co-occurrence Network ({topic})")
        show_graph_analytics(G)
        st.success(f"Analyzed {len(headlines)} headlines for topic '{topic}'.")
    if len(topic_sentiments) > 1:
        plot_topic_analytics(topic_sentiments)

st.markdown("---")
st.caption("Powered by Streamlit, HuggingFace Transformers, NewsAPI, NewsData.io, NetworkX, spaCy, and Plotly. Interactive graph analytics included.")


