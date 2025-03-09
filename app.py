import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from googletrans import Translator
import concurrent.futures

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load models and vectorizer
models = {
    "Logistic Regression": joblib.load('model_logistic_regression.pkl'),
    "Random Forest": joblib.load('model_random_forest.pkl'),
    "Gradient Boosting": joblib.load('model_gradient_boosting.pkl'),
    "XGBoost": joblib.load('model_xgboost.pkl')
}
vectorizer = joblib.load('vectorizer.pkl')

# Initialize translator
translator = Translator()

# Initialize PorterStemmer for English
port_stem = PorterStemmer()

# Function: Text summarization
def summarize_text(text, language="en", sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

# Function: Translate text
def translate_text(text, dest_language="en"):
    try:
        translated = translator.translate(text, src='auto', dest=dest_language)
        return translated.text
    except Exception as e:
        return f"Translation Error: {e}"

# Function: Analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

# Function: Predict with multiple models in parallel
def predict_multiple_models(transformed_text):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(model.predict, transformed_text): model_name for model_name, model in models.items()}
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            prediction = future.result()[0]
            confidence = models[model_name].predict_proba(transformed_text)[0][1]
            results[model_name] = {"prediction": prediction, "confidence": confidence}
    return results

# Streamlit UI setup
st.set_page_config(page_title="Fake News Prediction", page_icon="📰", layout="wide")

st.title("📰 Fake News Prediction")
st.write("Enter news details to check if it's **Fake News** or **Real News**.")

# Sidebar information
st.sidebar.title("About the App")
st.sidebar.info("This app predicts whether news is fake or real using multiple machine learning models.")
st.sidebar.write("Developed by: Rawinan Suwisut")

# Input fields
title = st.text_input("Enter News Title:")
author = st.text_input("Enter Author (Optional):")
text = st.text_area("Enter News Content:")

# Initialize session state for storing results
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
    st.session_state.summary_language = "English"
    st.session_state.translated_summary = ""
    st.session_state.sentiment = ""
    st.session_state.prediction_results = {}

# Submit button logic
if st.button("Submit"):
    if not title or not text:
        st.warning("⚠️ Please enter both title and content.")
    else:
        with st.spinner("Processing..."):
            full_text = f"{author} {title} {text}" if author else f"{title} {text}"
            summary_text = summarize_text(text, language="english")
            st.session_state.summary_text = summary_text
            st.session_state.translated_summary = translate_text(summary_text, dest_language="th")
            translated_text = translate_text(full_text, dest_language="en")
            processed_text = translated_text.lower()
            transformed_text = vectorizer.transform([processed_text])

            # Predict using multiple models
            st.session_state.prediction_results = predict_multiple_models(transformed_text)
            st.session_state.sentiment = analyze_sentiment(translated_text)
            
            st.success("✅ Processing complete! Scroll down to see the results.")

# Show Prediction Results
st.markdown("### 🔍 Prediction Results")
status = "❓ Waiting for input" if not st.session_state.prediction_results else ("❌ Fake News" if list(st.session_state.prediction_results.values())[0]["prediction"] == 1 else "✅ Real News")
st.write(f"**Prediction:** {status}")
st.write(f"**Sentiment Analysis:** {st.session_state.sentiment}")

# Show News Summary
st.markdown("### 🌐 News Summary")
sum_lang = st.selectbox("Choose summary language:", ["English", "Thai"], index=0 if st.session_state.summary_language == "English" else 1)

if sum_lang != st.session_state.summary_language:
    st.session_state.summary_language = sum_lang

summary = st.session_state.summary_text if st.session_state.summary_language == "English" else st.session_state.translated_summary
st.write(f"**News Summary ({st.session_state.summary_language}):** {summary}")

# View Details
with st.expander("View Details"):
    st.markdown("#### Model Predictions and Confidence")
    for model_name, result in st.session_state.prediction_results.items():
        model_status = "❌ Fake News" if result["prediction"] == 1 else "✅ Real News"
        st.write(f"**{model_name}:** {model_status} (Confidence: {result['confidence'] * 100:.2f}%)")
