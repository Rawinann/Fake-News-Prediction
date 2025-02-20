import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('train.csv')  # Dataset ของคุณ
data = data.dropna()  # ลบข้อมูลที่เป็นค่าว่าง

# Split data into input (X) and output (y)
X = data['text']
y = data['label']  # 1 = Fake, 0 = Real

# Text preprocessing with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

joblib.dump(model, 'model.pkl')  # บันทึกโมเดลลงไฟล์
print("Model saved successfully!")