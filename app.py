import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# Download stopwords
nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load('model.pkl')  # Logistic Regression model
vectorizer = joblib.load('vectorizer.pkl')  # TfidfVectorizer

# Define stemming function
port_stem = PorterStemmer()

def stemming(content):
    """Preprocess the input content by cleaning and stemming."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def extract_keywords(vectorizer, transformed_text):
    """Extract top keywords from the transformed text using TF-IDF weights."""
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = transformed_text.toarray()[0].argsort()[::-1]
    top_keywords = [feature_array[i] for i in tfidf_sorting[:5]]
    return top_keywords

def analyze_sentiment(text):
    """Analyze the sentiment of the input text."""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Streamlit app configuration
st.set_page_config(page_title="Fake News Prediction", page_icon="📰", layout="wide")

st.title("📰 Fake News Prediction")
st.write("Enter a news headline or content to check if it's **Fake News** or **Real News**.")

# Sidebar for additional information
st.sidebar.title("About the App")
st.sidebar.info(
    "This app predicts whether a piece of news is fake or real using a Logistic Regression model trained on a dataset of news articles. "
    "It processes text using TfidfVectorizer and NLTK for stemming and stopword removal."
)
st.sidebar.write("Developed by: Rawinan Suwisut")

# Input field for user
user_input = st.text_area("Enter the news content:", "")

if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to predict.")
    else:
        with st.spinner("Processing..."):
            # Preprocess and analyze input
            processed_text = stemming(user_input)
            transformed_text = vectorizer.transform([processed_text])
            prediction = model.predict(transformed_text)
            confidence = model.predict_proba(transformed_text)

            # Extract keywords
            keywords = extract_keywords(vectorizer, transformed_text)

            # Analyze sentiment
            sentiment = analyze_sentiment(user_input)

            # Display prediction result
            if prediction[0] == 1:
                st.error(f"❌ This news is predicted to be **Fake News**.")
            else:
                st.success(f"✅ This news is predicted to be **Real News**.")

            # Display additional information
            st.markdown("### 🔍 Analysis Details")
            st.write(f"**Confidence Score:** {confidence.max() * 100:.2f}%")
            st.write(f"**Top Keywords:** {', '.join(keywords)}")
            st.write(f"**Sentiment Analysis:** {sentiment}")


# Footer section
st.markdown("---")
st.markdown("### 📚 How does it work?")
st.write(
    """
    1. The app preprocesses the input text by removing non-alphabet characters, converting to lowercase, removing stopwords, and stemming.
    2. It uses a TfidfVectorizer to convert the text into numerical features.
    3. The Logistic Regression model predicts whether the text is **Fake News** or **Real News**.
    4. Additional analysis includes extracting top keywords and sentiment analysis.
    """
)
st.markdown("---")
st.write("Feel free to test the app with your own examples!")