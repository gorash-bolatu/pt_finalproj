from rest_framework import serializers
from .models import User, Product, Review, Sentiment, Recommendation


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'


class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = '__all__'


class SentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sentiment
        fields = '__all__'


class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = '__all__'
