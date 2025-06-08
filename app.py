import streamlit as st
st.set_page_config(page_title="Fake News Detector 🌐")

st.title("📰 Fake News Detector")
st.write("Model may take ~15 seconds the first time ⏳")

# --------- heavy libraries + setup ----------
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --------- run this only once, then cache result ----------
@st.cache_resource(show_spinner="Training model … this runs only once")
def load_and_train():
    nltk.download("stopwords", quiet=True)

    # Load data
    df = pd.read_csv("train.csv")
    df = df.fillna(" ")
    df["content"] = df.get("author", "") + " " + df.get("title", "")

    # Stemming
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    def stem_text(text):
        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower().split()
        return " ".join(ps.stem(word) for word in text if word not in stop_words)

    df["content"] = df["content"].apply(stem_text)

    # TF-IDF Vectorizer
    vector = TfidfVectorizer()
    X = vector.fit_transform(df["content"].values)
    y = df["label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, vector

# --------- run model training once & reuse ----------
model, vectorizer = load_and_train()

# --------- UI: Prediction ----------
def predict(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

user_input = st.text_input("Enter a news headline or article:")

if user_input:
    result = predict(user_input)
    if result == 1:
        st.error("❌ The news appears to be **FAKE**.")
    else:
        st.success("✅ The news appears to be **REAL**.")
