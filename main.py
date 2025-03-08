import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

nltk.download('stopwords')

# Load dataset
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake['class'] = 1
data_true['class'] = 0

data = pd.concat([data_fake, data_true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)  # Remove special characters
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

data['content'] = data['title'] + ' ' + data['text']
data['content'] = data['content'].apply(clean_text)

# Split data
X = data['content']
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert text to numeric vectors
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train models with optimized parameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    acc = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train_vect, y_train, cv=5).mean()
    print(f"{name} Accuracy: {acc}, Cross-Validation Score: {cv_score}")
    print(classification_report(y_test, y_pred))
    
    if cv_score > best_score:
        best_score = cv_score
        best_model = name
    
    filename = f"model_{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)

# Save vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
print(f"Best Model: {best_model} with CV Score: {best_score}")
print("All models and vectorizer saved.")