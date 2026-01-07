import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/train.csv")

# Clean questions
df['q1'] = df['question1'].apply(clean_text)
df['q2'] = df['question2'].apply(clean_text)

# Combine both questions
df['combined'] = df['q1'] + " " + df['q2']

X = df['combined']
y = df['is_duplicate']

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))

print("✅ Model training completed and saved!")
