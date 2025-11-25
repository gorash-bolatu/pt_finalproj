from django.urls import path
from .views import ReviewCreateView, analyze_sentiment, user_recommendations

urlpatterns = [
    path('review/create/', ReviewCreateView.as_view(), name='review-create'),
    path('sentiment/', analyze_sentiment, name='analyze-sentiment'),
    path('recommend/<int:user_id>/', user_recommendations, name='recommend'),
]
