# src/recommender.py
import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Determine models directory relative to this file (works when script is run from anywhere)
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(HERE, "..", "models"))

# Helpers to load artifacts safely
def _load_joblib(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required model artifact not found: {path}")
    return joblib.load(path)

def _load_pickle(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


# Load Naive Bayes artifacts
try:
    nb_model = _load_joblib("naive_bayes_model.joblib")
    nb_tfidf = _load_joblib("naive_bayes_tfidf.joblib")
    nb_label_enc = _load_joblib("naive_bayes_label_encoder.joblib")
except FileNotFoundError as e:
    # Re-raise with clearer message if you want, or keep None to allow partial operation
    raise

# Load SVM artifacts
try:
    svm_model = _load_joblib("svm_model.joblib")
    svm_tfidf = _load_joblib("svm_tfidf.joblib")
    svm_label_enc = _load_joblib("svm_label_encoder.joblib")
except FileNotFoundError as e:
    raise

# Load LSTM artifacts
# training saved `lstm_model.h5` (or .h5) and tokenizer pickle and label encoder joblib
lstm_model_path_h5 = os.path.join(MODELS_DIR, "lstm_model.h5")
if not os.path.exists(lstm_model_path_h5):
    # fall back to commonly used alternative filename (if any)
    lstm_model_path_h5 = os.path.join(MODELS_DIR, "lstm_model.keras")

if not os.path.exists(lstm_model_path_h5):
    raise FileNotFoundError(f"LSTM model file not found (checked .h5 and .keras) in {MODELS_DIR}")

lstm_model = tf.keras.models.load_model(lstm_model_path_h5)
lstm_tokenizer = _load_pickle("lstm_tokenizer.pkl")
lstm_label_enc = _load_joblib("lstm_label_encoder.joblib")

# optionally read config for LSTM max_len
lstm_cfg = _load_json("lstm_config.json") or {}
LSTM_MAX_LEN = int(lstm_cfg.get("max_len", 200))


# Prediction function
def predict_sentiment(text: str, model_name: str = "nb"):
    """
    Predict sentiment using NB, SVM, or LSTM.
    Returns the label string (e.g., 'positive'/'neutral'/'negative').
    """
    model_name = model_name.lower()

    if model_name == "nb":
        X = nb_tfidf.transform([text])
        pred_enc = nb_model.predict(X)[0]
        # invert encoded label to string
        try:
            label = nb_label_enc.inverse_transform([pred_enc])[0]
        except Exception:
            # if label encoder isn't available or mapping is numeric
            label = str(pred_enc)

    elif model_name == "svm":
        X = svm_tfidf.transform([text])
        pred_enc = svm_model.predict(X)[0]
        try:
            label = svm_label_enc.inverse_transform([pred_enc])[0]
        except Exception:
            label = str(pred_enc)

    elif model_name == "lstm":
        seq = lstm_tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=LSTM_MAX_LEN, padding="post", truncating="post")
        probs = lstm_model.predict(seq, verbose=0)
        # If binary output of shape (1, ) or (1,1) interpret sigmoid; else argmax
        if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
            prob = float(probs.reshape(-1)[0])
            pred_enc = 1 if prob > 0.5 else 0
        else:
            pred_enc = int(np.argmax(probs, axis=1)[0])

        try:
            label = lstm_label_enc.inverse_transform([pred_enc])[0]
        except Exception:
            label = str(pred_enc)

    else:
        raise ValueError("Unknown model_name. Use 'nb', 'svm', or 'lstm'.")

    return label


# Recommender
def recommend_products(user_text: str, product_db: pd.DataFrame, model_name: str = "svm"):
    """
    Recommend products based on:
      - predicted sentiment of user_text
      - product_db: pandas DataFrame with columns 'product_id', 'review_text', 'sentiment'
    product_db sentiment column may contain numeric flags (1/0) or strings ('positive', 'negative').
    Returns a dict: {"user_sentiment": label, "recommended_products": [product_ids]}
    """
    if not isinstance(product_db, pd.DataFrame):
        raise TypeError("product_db must be a pandas DataFrame")

    user_sentiment = predict_sentiment(user_text, model_name=model_name)

    # Normalize sentiment column for matching: accept numeric 1, 'positive', 'pos'
    def is_positive_val(v):
        if pd.isna(v):
            return False
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(v) == 1
        sv = str(v).strip().lower()
        return sv in {"positive", "pos", "1", "true", "yes"}

    # Filter positive reviews
    try:
        positives = product_db[product_db["sentiment"].apply(is_positive_val)]
    except Exception:
        # if column missing or unusual, default to empty
        positives = product_db.iloc[0:0]

    if user_sentiment and str(user_sentiment).strip().lower() in {"positive", "pos"}:
        matches = positives
    else:
        # For negative users, recommend popular positive products (sample if needed)
        matches = positives.sample(n=min(100, len(positives))) if len(positives) > 0 else positives

    if matches.empty:
        return {"user_sentiment": user_sentiment, "recommended_products": []}

    # Recommend products with highest number of positive reviews
    recommendations = (
        matches.groupby("product_id")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    return {"user_sentiment": user_sentiment, "recommended_products": recommendations}


# Example usage
if __name__ == "__main__":
    # quick local smoke test
    sample = "This product is amazing and works perfectly!"
    print("NB:", predict_sentiment(sample, "nb"))
    print("SVM:", predict_sentiment(sample, "svm"))
    print("LSTM:", predict_sentiment(sample, "lstm"))

    # example recommender (toy)
    df = pd.DataFrame({
        "product_id": ["p1","p1","p2","p3","p1","p2"],
        "review_text": ["Good","Excellent","Nice","Bad","Loved it","Good"],
        "sentiment": [1,1,1,0,1,1]
    })
    print(recommend_products(sample, df, model_name="svm"))
