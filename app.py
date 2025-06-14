import streamlit as st
import re
import nltk
import joblib

# Download stopwords
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return " ".join(text.split())

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
user_input = st.text_area("📝 Paste your news content here:")

if st.button("Check"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        if prediction[0] == 1:
            st.success("✅ This appears to be **REAL** news.")
        else:
            st.error("🚨 This appears to be **FAKE** news.")
    else:
        st.warning("Please enter some text.")
