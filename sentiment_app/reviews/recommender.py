# reviews/recommender.py

import joblib
import tensorflow as tf
import pandas as pd
from reviews.ml_predict import predict_sentiment

# Load models and transformers
# NB and SVM use the same TF-IDF vectorizer
vectorizer = joblib.load(r"reviews\models\naive_bayes_tfidf.joblib")  # shared for NB/SVM

# LSTM
lstm_model = tf.keras.models.load_model(r"reviews\models\lstm_model.h5")
with open(r"reviews\models\lstm_tokenizer.pkl", "rb") as f:
    lstm_tokenizer = joblib.load(f)


def recommend_products(user_text: str, product_df: pd.DataFrame, ml_model="svm"):
    """
    Returns top 5 product IDs to recommend based on user's review text.

    Parameters:
        user_text: str
        product_df: pandas DataFrame with columns ["product_id", "review_text", "sentiment"]
        ml_model: "nb", "svm", or "lstm" (controls which ML model to use for sentiment)

    Returns:
        dict with keys:
            - "user_sentiment": str ("positive" or "negative")
            - "recommended_products": list of top 5 product_ids
    """

    # 1) Predict sentiment
    user_sentiment = predict_sentiment(user_text, model_name=ml_model)

    # 2) Filter products based on sentiment
    if user_sentiment == "positive":
        filtered = product_df[product_df["sentiment"] == "positive"]
    else:
        # for negative users, pick a mix of products randomly
        filtered = product_df.sample(min(len(product_df), 50))

    # 3) Rank products by number of positive reviews
    recommendations = (
        filtered.groupby("product_id")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    return {"user_sentiment": user_sentiment, "recommended_products": recommendations}


def format_for_django(recommendation_dict, product_model):
    """
    Converts product IDs into actual Product queryset for template rendering.
    """
    product_ids = recommendation_dict.get("recommended_products", [])
    products = product_model.objects.filter(id__in=product_ids)
    return products
