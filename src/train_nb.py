import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Config 
DATA_PATH = os.path.join("data", "processed", "train_clean.csv")
OUT_DIR = os.path.join("models")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MAX_SAMPLES = 20000   # reduce to control memory/time (set to None to use all)
TEST_SIZE = 0.1

TFIDF_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),
}

MODEL_NAME = "naive_bayes"

# Load data 
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'sentiment'])
if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)

X = df['text'].astype(str).values
y = df['sentiment'].astype(str).values

# Encode labels 
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split 
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)

# Vectorize 
tfidf = TfidfVectorizer(**TFIDF_PARAMS)
Xtr = tfidf.fit_transform(X_train)
Xv = tfidf.transform(X_val)

# Train 
clf = MultinomialNB()
clf.fit(Xtr, y_train)

# Eval 
y_pred = clf.predict(Xv)
report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)

# Save artifacts 
joblib.dump(clf, os.path.join(OUT_DIR, f"{MODEL_NAME}_model.joblib"))
joblib.dump(tfidf, os.path.join(OUT_DIR, f"{MODEL_NAME}_tfidf.joblib"))
joblib.dump(le, os.path.join(OUT_DIR, f"{MODEL_NAME}_label_encoder.joblib"))

config = {
    "model": MODEL_NAME,
    "tfidf_params": TFIDF_PARAMS,
    "random_state": RANDOM_STATE,
    "max_samples": MAX_SAMPLES,
    "test_size": TEST_SIZE
}
with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_config.json"), "w", encoding="utf8") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_metrics.json"), "w", encoding="utf8") as f:
    json.dump(report, f, indent=2)

print("Naive Bayes training complete. Artifacts saved to", OUT_DIR)
print("Sample evaluation (macro avg):", report.get("macro avg", {}))
