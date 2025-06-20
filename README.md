# Daily Dose

**Daily Dose** is a public repository demonstrating a collection of real-time AI-powered applications focused on Natural Language Processing, financial analytics, and predictive modeling.

Designed as a showcase of applied skills in AI, data science, and ML engineering, this project includes end-to-end pipelines involving machine learning, deep learning, sentiment analysis, time-series forecasting, and real-time financial data integration — all built with Python, Streamlit, and public data sources (with some datasets realistically simulated when access was limited or commercially restricted).

More modules, use cases, and technical documentation will be added as the project evolves.

## Currently Deployed

### 🔹 Crypto News Sentiment & Price Analyzer

This module explores the relationship between cryptocurrency news sentiment and market prices using real-time RSS feeds, sentiment scoring (VADER), and price data via Yahoo Finance.

Users can:
- View sentiment trends from real crypto news
- Analyze live price data
- Measure correlation between market mood and price changes
- Upload their own news articles for summarization and sentiment evaluation

**Live App:** [https://crypto-sentiment-analyzer.streamlit.app](https://crypto-sentiment-analyzer.streamlit.app)

---

### 🔹 Multi-Source News Sentiment & Graph Analytics

This module aggregates headlines across multiple sources (NewsAPI, NewsData.io, BBC, Reuters) and performs real-time sentiment analysis, named entity recognition (NER), and graph-based relationship mapping.

Features include:
- Sentiment analysis using Hugging Face Transformers
- Real-time headline fetching and deduplication
- Named entity extraction via spaCy
- Entity co-occurrence network graphs with centrality insights
- Topic-wise comparison and visualization of sentiment trends

**Live App:** [https://news-sentiment-relationship-analyzer.streamlit.app](https://news-sentiment-relationship-analyzer.streamlit.app)

---

### 🔹 News Category Analyzer (LLM-powered)

This tool fetches real-time news articles using NewsAPI and uses OpenAI GPT to:
- Summarize each article in 2–3 sentences  
- Categorize it into predefined domains (Technology, Finance, Science, etc.)

Use cases include:
- Rapid news digestion
- Trend analysis by domain
- AI-powered classification demos

**Live App:** [https://news-category-analyzer.streamlit.app](https://news-category-analyzer.streamlit.app)

---

## License

This project is licensed under the **Apache License 2.0**. See [`LICENSE`](./LICENSE) [`NOTICE`](./NOTICE) for details.

---

## About

Developed and maintained by Sridhar Somisetti as part of an independent research and development effort in applied AI/ML systems.

## Contact

- GitHub: [Somisetti-Sridhar](https://github.com/Somisetti-Sridhar)
- LinkedIn: [linkedin.com/in/-sridhar-](https://linkedin.com/in/-sridhar-)
