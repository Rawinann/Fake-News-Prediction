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
import matplotlib.pyplot as plt
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

# Thai stopwords
THAI_STOPWORDS = [
    "ที่", "และ", "คือ", "ไม่", "ใน", "ให้", "ได้", "โดย", "จะ", "มี", "ของ", "ได้", "จาก", "เป็น", "ว่า", "ซึ่ง",
    "ของ", "กับ", "อีก", "นี้", "นั้น", "อัน", "หนึ่ง", "สอง", "สาม", "เป็น", "เป็นต้น", "หรือ", "กับ", "ถึง"
]

# Function: Tokenize Thai text using regex
def custom_tokenize_thai(text):
    tokens = re.findall(r'\w+', text)
    return tokens

# Function: Remove Thai stopwords
def remove_thai_stopwords(tokens):
    return [token for token in tokens if token not in THAI_STOPWORDS]

# Function: Preprocess Thai text
def preprocess_thai(content):
    tokens = custom_tokenize_thai(content)
    cleaned_tokens = remove_thai_stopwords(tokens)
    return ' '.join(cleaned_tokens)

# Function: Preprocess English text
def preprocess_english(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Function: Text summarization
def summarize_text(text, language="en", sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

# Function: Analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

# Function: Translate text
def translate_text(text, dest_language="en"):
    if not text:
        return ""
    try:
        translated = translator.translate(text, src='auto', dest=dest_language)
        return translated.text if translated and translated.text else text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to collect user feedback
def collect_user_feedback():
    feedback = st.radio("Do you think this news is real or fake?", ("Real", "Fake"))
    if feedback:
        # Save user feedback for future use (For example, save to a database or a file)
        st.write(f"Feedback received: {feedback}")
        return feedback
    return None

# Function to plot confidence scores
def plot_confidence(results):
    models = list(results.keys())
    confidences = [result["confidence"] for result in results.values()]
    
    plt.figure(figsize=(10, 5))
    plt.bar(models, confidences, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Confidence')
    plt.title('Confidence Scores for Different Models')
    plt.ylim(0, 1)
    st.pyplot(plt)

# Function to predict with multiple models in parallel
def predict_multiple_models(transformed_text):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for model_name, model in models.items():
            futures[executor.submit(model.predict, transformed_text)] = model_name
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            prediction = future.result()[0]
            confidence = model.predict_proba(transformed_text)[0][1]
            results[model_name] = {"prediction": prediction, "confidence": confidence}
    return results

# Streamlit UI setup
st.set_page_config(page_title="Fake News Prediction", page_icon="📰", layout="wide")

st.title("📰 Fake News Prediction")
st.write("Enter news details to check if it's **Fake News** or **Real News**.")

# Sidebar information
st.sidebar.title("About the App")
st.sidebar.info(
    "This app predicts whether news is fake or real using multiple machine learning models. It also provides sentiment analysis and news summarization."
)
st.sidebar.write("Developed by: Rawinan Suwisut")

# Input fields
title = st.text_input("Enter News Title:")
author = st.text_input("Enter Author (Optional):")
text = st.text_area("Enter News Content:")

# Submit button logic
if st.button("Submit"):
    if not title or not text:
        st.warning("⚠️ Please enter both title and content.")
    else:
        with st.spinner("Processing..."):
            try:
                full_text = f"{author} {title} {text}" if author else f"{title} {text}"
                translated_text = translate_text(full_text, dest_language="en")
                processed_text = preprocess_english(translated_text)
                transformed_text = vectorizer.transform([processed_text])
                
                # Predict using multiple models in parallel
                results = predict_multiple_models(transformed_text)
                
                sentiment = analyze_sentiment(translated_text)
                summary_text = summarize_text(translated_text, language="english")
                
                # Display confidence graph
                plot_confidence(results)
                
                # Display results
                st.markdown("### 🔍 Analysis Details")
                for model_name, result in results.items():
                    status = "❌ Fake News" if result["prediction"] == 1 else "✅ Real News"
                    st.write(f"**{model_name}:** {status} (Confidence: {result['confidence'] * 100:.2f}%)")
                    
                    with st.expander(f"View Details ({model_name})"):
                        st.write(f"**Prediction:** {status}")
                        st.write(f"**Confidence Score:** {result['confidence'] * 100:.2f}%")
                        st.write(f"**Sentiment Analysis:** {sentiment}")
                        summary_language = st.selectbox(f"Choose summary language ({model_name}):", ["English", "Thai"], key=model_name)
                        summary = translate_text(summary_text, dest_language="th") if summary_language == "Thai" else summary_text
                        st.write(f"**News Summary ({summary_language}):** {summary}")
                
                # Collect feedback
                feedback = collect_user_feedback()
                if feedback:
                    # Save or process feedback (For future improvements)
                    st.write(f"Thank you for your feedback! We will use it to improve our predictions.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
