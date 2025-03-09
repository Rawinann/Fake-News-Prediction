import streamlit as st
import joblib
import re
import pandas as pd
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

# Function: Translate text safely
def translate_text(text, dest_language="en"):
    if not text:
        return ""
    try:
        translated = translator.translate(text, src='auto', dest=dest_language)
        return translated.text if translated and translated.text is not None else text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

# Function: Predict with multiple models
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

# Store the language selection in session_state only once
if 'summary_language' not in st.session_state:
    st.session_state.summary_language = "English"

# Submit button
if st.button("Submit"):
    if not title or not text:
        st.warning("⚠️ Please enter both title and content.")
    else:
        with st.spinner("Processing..."):
            full_text = f"{author} {title} {text}" if author else f"{title} {text}"
            translated_text = translate_text(full_text, dest_language="en")

            # If translation fails, fallback to original text
            processed_text = translated_text if translated_text else full_text
            transformed_text = vectorizer.transform([processed_text])

            # Predict using multiple models
            results = predict_multiple_models(transformed_text)
            sentiment = analyze_sentiment(translated_text)
            summary_text = summarize_text(translated_text, language="english")

            # Display results
            status = "❌ Fake News" if results["Logistic Regression"]["prediction"] == 1 else "✅ Real News"
            st.markdown("### 🔍 Prediction Results")
            st.write(f"**Prediction:** {status}")
            st.write(f"**Sentiment Analysis:** {sentiment}")

            # Language selection for summary
            summary_language = st.selectbox("Choose summary language:", ["English", "Thai"], index=["English", "Thai"].index(st.session_state.summary_language))
            st.session_state.summary_language = summary_language

            # Translate summary only if necessary
            summary = translate_text(summary_text, dest_language="th") if summary_language == "Thai" else summary_text
            st.write(f"**News Summary ({summary_language}):** {summary}")

            # View Details
            with st.expander("View Details"):
                confidence_data = []

                for model_name, result in results.items():
                    model_status = "❌ Fake News" if result["prediction"] == 1 else "✅ Real News"
                    confidence_percentage = result["confidence"] * 100

                    st.write(f"**{model_name}:** {model_status} (Confidence: {confidence_percentage:.2f}%)")

                    # Store confidence data for chart
                    confidence_data.append({"Model": model_name, "Confidence": confidence_percentage})

                # Convert to DataFrame for plotting
                if confidence_data:
                    df_confidence = pd.DataFrame(confidence_data)
                    st.line_chart(df_confidence.set_index("Model"))  # Use index to show model names correctly