from django.shortcuts import render
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import User, Product, Review, Sentiment, Recommendation
from .serializers import (
    UserSerializer, ProductSerializer, ReviewSerializer,
    SentimentSerializer, RecommendationSerializer
)

import joblib


# Load ML model on first import
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")


class ReviewCreateView(generics.CreateAPIView):
    queryset = Review.objects.all()
    serializer_class = ReviewSerializer


@api_view(["POST"])
def analyze_sentiment(request):
    text = request.data.get("text")
    if not text:
        return Response({"error": "No text provided"}, status=400)

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    return Response({"sentiment": pred})


@api_view(["GET"])
def user_recommendations(request, user_id):
    recs = Recommendation.objects.filter(user_id=user_id)
    serializer = RecommendationSerializer(recs, many=True)
    return Response(serializer.data)

