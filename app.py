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

# Thai stopwords (custom list)
THAI_STOPWORDS = [
    "ที่", "และ", "คือ", "ไม่", "ใน", "ให้", "ได้", "โดย", "จะ", "มี", "ของ", "ได้", "จาก", "เป็น", "ว่า", "ซึ่ง"
]

# Function: Tokenize Thai text using regex
def custom_tokenize_thai(text):
    """Tokenize Thai text using regular expressions."""
    tokens = re.findall(r'\w+', text)
    return tokens

# Function: Remove Thai stopwords
def remove_thai_stopwords(tokens):
    """Remove Thai stopwords from the tokenized text."""
    return [token for token in tokens if token not in THAI_STOPWORDS]

# Function: Preprocess Thai text
def preprocess_thai(content):
    """Preprocess Thai content by removing Thai stopwords."""
    tokens = custom_tokenize_thai(content)
    cleaned_tokens = remove_thai_stopwords(tokens)
    return ' '.join(cleaned_tokens)

# Function: Preprocess English text
def preprocess_english(content):
    """Clean and preprocess English content using stemming and stopword removal."""
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Function: Text summarization
def summarize_text(text, language="en", sentences_count=2):
    """Summarize the input text in the specified language."""
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

# Function: Analyze sentiment
def analyze_sentiment(text):
    """Analyze the sentiment of the input text."""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

# Function: Translate text
def translate_text(text, dest_language="en"):
    """Translate text to the specified destination language (default: English)."""
    translated = translator.translate(text, src='auto', dest=dest_language)
    return translated.text

# Streamlit UI setup
st.set_page_config(page_title="Fake News Prediction", page_icon="📰", layout="wide")

st.title("📰 Fake News Prediction")
st.write("Enter a news headline or content to check if it's **Fake News** or **Real News**.")

# Sidebar information
st.sidebar.title("About the App")
st.sidebar.info(
    "This app predicts whether news is fake or real using a Logistic Regression model. "
    "It supports multilingual processing, sentiment analysis, keyword extraction, and summarization."
)
st.sidebar.write("Developed by: Rawinan Suwisut")

# Initialize session state for storing results
if "processed_text" not in st.session_state:
    st.session_state.processed_text = None
    st.session_state.translated_text = None
    st.session_state.summary_text = None
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.sentiment = None

# User input
user_input = st.text_area("Enter the news content:", "")

if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to predict.")
    else:
        with st.spinner("Processing..."):
            try:
                # Translate text to English for processing
                translated_text = translate_text(user_input, dest_language="en")

                # Preprocess text based on language
                processed_text = preprocess_english(translated_text)

                # Transform text using vectorizer
                transformed_text = vectorizer.transform([processed_text])

                # Predict fake or real news
                prediction = model.predict(transformed_text)
                confidence = model.predict_proba(transformed_text)

                # Perform sentiment analysis
                sentiment = analyze_sentiment(translated_text)

                # Summarize the text
                summary_text = summarize_text(translated_text, language="english")

                # Store results in session state
                st.session_state.translated_text = translated_text
                st.session_state.processed_text = processed_text
                st.session_state.summary_text = summary_text
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.sentiment = sentiment

                st.success("✅ Processing complete! Scroll down to see the results.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display results if processed
if st.session_state.processed_text:
    st.markdown("### 🔍 Analysis Details")

    # Show prediction results
    if st.session_state.prediction[0] == 1:
        st.error(f"❌ This news is predicted to be **Fake News**.")
    else:
        st.success(f"✅ This news is predicted to be **Real News**.")

    st.write(f"**Confidence Score:** {st.session_state.confidence.max() * 100:.2f}%")
    st.write(f"**Sentiment Analysis:** {st.session_state.sentiment}")

    # Choose summary language dynamically
    summary_language = st.selectbox("Choose summary language:", ["English", "Thai"])
    if summary_language == "Thai":
        summary = translate_text(st.session_state.summary_text, dest_language="th")
    else:
        summary = st.session_state.summary_text

    st.write(f"**News Summary ({summary_language}):** {summary}")