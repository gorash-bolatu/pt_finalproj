import pytest
import pandas as pd
from src.recommender import predict_sentiment, recommend_products

def test_predict_sentiment_nb():
    text = "I love this product!"
    pred = predict_sentiment(text, model_name="nb")
    assert pred in ["positive", "neutral", "negative"]

def test_predict_sentiment_svm():
    text = "Terrible experience."
    pred = predict_sentiment(text, model_name="svm")
    assert pred in ["positive", "neutral", "negative"]

def test_predict_sentiment_lstm():
    text = "Amazing quality!"
    pred = predict_sentiment(text, model_name="lstm")
    assert pred in ["positive", "neutral", "negative"]

def test_recommend_products():
    df = pd.DataFrame({
        "product_id": ["p1","p2","p3","p1"],
        "review_text": ["good","bad","good","excellent"],
        "sentiment": [1,0,1,1]
    })
    result = recommend_products("I like this!", df, model_name="svm")
    assert "user_sentiment" in result
    assert "recommended_products" in result
    assert isinstance(result["recommended_products"], list)
