import pandas as pd
import pytest

# Import the functions from your recommender module.
# The test assumes `recommend_products(product_db, ...)` is available and predict_sentiment is used internally.
from src import recommender


def test_recommend_products_positive(monkeypatch):
    # Create a tiny product_db DataFrame
    df = pd.DataFrame({
        "product_id": ["p1","p1","p2","p3","p1","p2"],
        "review_text": [
            "Good","Excellent","Nice","Bad","Loved it","Good"
        ],
        "sentiment": [1,1,1,0,1,1]  # 1 = positive
    })

    # Monkeypatch predict_sentiment to always return 'positive'
    monkeypatch.setattr(recommender, "predict_sentiment", lambda text, model_name="svm": "positive")

    out = recommender.recommend_products("I like it", df)
    assert out["user_sentiment"] == "positive"
    assert isinstance(out["recommended_products"], list)
    # p1 should be top since it has three positive reviews
    assert "p1" in out["recommended_products"]


def test_recommend_products_negative(monkeypatch):
    df = pd.DataFrame({
        "product_id": ["p1","p2","p3","p4"],
        "review_text": ["Bad","Terrible","Okay","Good"],
        "sentiment": [0,0,1,1]
    })
    # For negative users we sample positives (see recommender logic)
    monkeypatch.setattr(recommender, "predict_sentiment", lambda text, model_name="svm": "negative")

    out = recommender.recommend_products("I hate this", df)
    assert out["user_sentiment"] == "negative"
    assert isinstance(out["recommended_products"], list)
