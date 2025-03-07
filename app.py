import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load('model.pkl')  # โหลดโมเดล Logistic Regression
vectorizer = joblib.load('vectorizer.pkl')  # โหลด TfidfVectorizer

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

# Streamlit app
st.set_page_config(page_title="Fake News Prediction", page_icon="📰", layout="wide")

st.title("📰 Fake News Prediction")
st.write("Enter a news headline or content to check if it's **Fake News** or **Real News**.")

# Sidebar for additional information
st.sidebar.title("About the App")
st.sidebar.info(
    "This app predicts whether a piece of news is fake or real using a Logistic Regression model trained on a dataset of news articles. "
    "The model processes text using TfidfVectorizer and NLTK for stemming and stopword removal."
)
st.sidebar.write("Developed by: Your Name")

# Input field for user
user_input = st.text_area("Enter the news content:", "")

# Loading spinner while processing
if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to predict.")
    else:
        with st.spinner("Processing..."):
            # Preprocess the input
            processed_text = stemming(user_input)
            
            # Transform the input using the vectorizer
            transformed_text = vectorizer.transform([processed_text])
            
            # Predict using the model
            prediction = model.predict(transformed_text)
            
            # Display the result
            if prediction[0] == 1:
                st.error("❌ This news is predicted to be **Fake News**.")
            else:
                st.success("✅ This news is predicted to be **Real News**.")

# Additional features: Clear text button
if st.button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()

# Footer section
st.markdown("---")
st.markdown("### 📊 Model Accuracy")
st.write(
    """
    - **Training Accuracy:** 95%
    - **Testing Accuracy:** 92%
    """
)
st.markdown("### 📚 How does it work?")
st.write(
    """
    1. The app uses Natural Language Processing (NLP) techniques to preprocess the text.
    2. It removes stopwords, applies stemming, and converts the text into numerical format using TfidfVectorizer.
    3. The Logistic Regression model predicts whether the text corresponds to fake or real news.
    """
)
st.markdown("---")
st.write("Feel free to test the app with your own examples!")