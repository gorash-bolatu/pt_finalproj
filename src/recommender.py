import pickle
import numpy as np
import tensorflow as tf

# LOAD MODELS AND TRANSFORMERS

# Traditional ML models
with open("models/nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# LSTM model + tokenizer
lstm_model = tf.keras.models.load_model("models/lstm_model.keras")

with open("models/lstm_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# SENTIMENT INFERENCE

def predict_sentiment(text: str, model_name: str = "nb"):
    """
    Predict sentiment using NB, SVM, or LSTM.
    Returns: label 'positive'/'negative'
    """

    if model_name.lower() == "nb":
        X = vectorizer.transform([text])
        pred = nb_model.predict(X)[0]

    elif model_name.lower() == "svm":
        X = vectorizer.transform([text])
        pred = svm_model.predict(X)[0]

    elif model_name.lower() == "lstm":
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
        prob = lstm_model.predict(padded, verbose=0)[0][0]
        pred = 1 if prob > 0.5 else 0

    else:
        raise ValueError("Unknown model name. Use 'nb', 'svm', or 'lstm'.")

    return "positive" if pred == 1 else "negative"


# SIMPLE RECOMMENDER

def recommend_products(user_text: str, product_db):
    """
    Recommend products based on:
    - predicted sentiment
    - similar reviews from product_db (pandas DataFrame)
    
    product_db must have columns:
        'product_id', 'review_text', 'sentiment'

    Returns a list of product IDs.
    """

    user_sentiment = predict_sentiment(user_text, model_name="svm")

    if user_sentiment == "positive":
        # Recommend products liked by others
        matches = product_db[product_db["sentiment"] == 1]
    else:
        # Recommend opposite sentiment
        matches = product_db[product_db["sentiment"] == 1].sample(10)

    # Simple rule: recommend the products with highest number of positive reviews
    recommendations = (
        matches.groupby("product_id")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    return {
        "user_sentiment": user_sentiment,
        "recommended_products": recommendations,
    }