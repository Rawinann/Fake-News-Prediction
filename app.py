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

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load model and vectorizer
model = joblib.load('model.pkl')
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
    if not text:  # ถ้าไม่มีข้อความ ไม่ต้องแปล
        return ""
    try:
        # แปลภาษาไทยไปเป็นอังกฤษ
        translated = translator.translate(text, src='auto', dest=dest_language)
        return translated.text if translated and translated.text else text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # ถ้าเกิดข้อผิดพลาด ให้คืนค่าข้อความต้นฉบับ

# Streamlit UI setup
st.set_page_config(page_title="Fake News Prediction", page_icon="📰", layout="wide")

st.title("📰 Fake News Prediction")
st.write("Enter news details to check if it's **Fake News** or **Real News**.")

# Sidebar information
st.sidebar.title("About the App")
st.sidebar.info(
    "This app predicts whether news is fake or real using a machine learning model. It also provides sentiment analysis and news summarization.")
st.sidebar.write("Developed by: Rawinan Suwisut")

# Initialize session state
if "submitted" not in st.session_state:
    st.session_state.submitted = False
    st.session_state.title = ""
    st.session_state.author = ""
    st.session_state.text = ""

# Input fields
title = st.text_input("Enter News Title:", st.session_state.title)
author = st.text_input("Enter Author (Optional):", st.session_state.author)
text = st.text_area("Enter News Content:", st.session_state.text)

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
                prediction = model.predict(transformed_text)
                confidence = model.predict_proba(transformed_text)
                sentiment = analyze_sentiment(translated_text)
                summary_text = summarize_text(translated_text, language="english")

                if author and title:
                    confidence[0][1] += 0.05
                elif author:
                    confidence[0][1] += 0.04
                elif title:
                    confidence[0][1] += 0.03
                confidence[0][1] = min(confidence[0][1], 1.0)

                st.session_state.submitted = True
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.sentiment = sentiment
                st.session_state.summary_text = summary_text
                st.session_state.title = title
                st.session_state.author = author
                st.session_state.text = text

                st.rerun()  # Rerun the app to refresh and keep 'Submit' button visible
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display result only if submitted
if st.session_state.submitted:
    st.markdown("### 🔍 Analysis Details")
    if st.session_state.prediction[0] == 1:
        st.error("❌ This news is predicted to be **Fake News**.")
    else:
        st.success("✅ This news is predicted to be **Real News**.")

    st.write(f"**Confidence Score:** {st.session_state.confidence.max() * 100:.2f}%")
    st.write(f"**Sentiment Analysis:** {st.session_state.sentiment}")

    summary_language = st.selectbox("Choose summary language:", ["English", "Thai"])
    summary = translate_text(st.session_state.summary_text, dest_language="th") if summary_language == "Thai" else st.session_state.summary_text
    st.write(f"**News Summary ({summary_language}):** {summary}")

    # Button to clear all session state and reset form for next use
    if st.button("Clear"):
        st.session_state.submitted = False
        st.session_state.title = ""
        st.session_state.author = ""
        st.session_state.text = ""
        st.rerun()  # Rerun to refresh the UI and allow for new data input
