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

print(X)
print(Y)
Y.shape