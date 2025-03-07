import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

nltk.download('stopwords')
# print(stopwords.words('english'))

# Load dataset
data = pd.read_csv('train.csv')  # Dataset ของคุณ

data.isnull().sum()
data = data.fillna('')  # ลบข้อมูลที่เป็นค่าว่าง

# merging the author name and news title
data['content'] = data['author']+' '+data['title']
print(data['content'])

# Split data into input (X) and output (y)
X = data.drop(columns='label', axis=1)
Y = data['label']

port_stem = PorterStemmer()

data['content'] = data['content'].apply(stemming)
# print(data['content'])

#separating the data and label
X = data['content'].values
Y = data['label'].values
# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# Save the model as a pickle file
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")