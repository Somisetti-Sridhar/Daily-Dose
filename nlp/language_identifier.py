import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

@st.cache_data
def load_data():
    # Source: https://www.statmt.org/europarl/
    languages = ['en', 'fr', 'de', 'es', 'it']
    data = []
    
    for lang in languages:
        with open(f'data/{lang}.txt', 'r', encoding='utf-8') as f:
            text = f.read().splitlines()[:1000]
            data.extend([(line, lang) for line in text])
    
    return pd.DataFrame(data, columns=['text', 'language'])

@st.cache_resource
def train_model(data):
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) 
        text = ' '.join([word for word in text.split() 
                        if word not in stopwords.words()])
        return text
    data['cleaned_text'] = data['text'].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], data['language'], test_size=0.2, random_state=42
    )

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

st.title('Multilingual Language Detection App')
st.write("""
Detect the language of text using machine learning. This model was trained on European Parliament proceedings data in:
- English (en)
- French (fr)
- German (de)
- Spanish (es)
- Italian (it)
""")

data = load_data()
model, accuracy = train_model(data)

st.sidebar.header('Model Information')
st.sidebar.metric("Test Accuracy", f"{accuracy:.1%}")
st.sidebar.write(f"Trained on {len(data)} samples")


user_input = st.text_area("Enter text for language detection:", 
                          "Bonjour! Comment allez-vous aujourd'hui?")

if st.button('Detect Language'):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'it': 'Italian'
        }
        
        st.success(f"Predicted language: **{lang_names[prediction]}**")
        st.write("### Prediction Confidence:")
        
        probs = model.predict_proba([user_input])[0]
        lang_prob = dict(zip(model.classes_, probs))
        
        for lang, prob in lang_prob.items():
            st.progress(prob, text=f"{lang_names[lang]}: {prob:.1%}")
    else:
        st.warning("Please enter some text")

if st.checkbox("Show training data samples"):
    st.write("### Sample Data from European Parliament Proceedings")
    st.dataframe(data.sample(10, random_state=42))
