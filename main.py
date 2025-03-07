import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define stemming function
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Streamlit app
st.title("Fake News Prediction")
st.write("Enter a news headline or content to check if it's fake or real")

# User input
user_input = st.text_area("Enter the news content:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter some text to predict.")
    else:
        # Preprocess the input
        processed_text = stemming(user_input)
        
        # Transform the input using the vectorizer
        transformed_text = vectorizer.transform([processed_text])
        
        # Predict using the model
        prediction = model.predict(transformed_text)
        
        # Display the result
        if prediction[0] == 1:
            st.error("This news is predicted to be **Fake News**.")
        else:
            st.success("This news is predicted to be **Real News**.")