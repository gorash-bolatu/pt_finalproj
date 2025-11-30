# reviews/views.py
from django.shortcuts import render
from .ml import analyze_sentiment
from .models import Product

def product_list(request):
    products = Product.objects.all()
    return render(request, "products/list.html", {"products": products})

def recommend_view(request):
    return render(request, "recommendations/result.html", {})

def home(request):
    result = None
    user_text = ""
    model_choice = "nb"  # default model

    if request.method == "POST":
        user_text = request.POST.get("review_text", "")
        model_choice = request.POST.get("model_choice", "nb")

        if user_text.strip():
            result = analyze_sentiment(user_text, model_type=model_choice)

    context = {
        "result": result,
        "user_text": user_text,
        "model_choice": model_choice,
        "available_models": [
            ("nb", "Naive Bayes"),
            ("svm", "SVM"),
            ("lstm", "LSTM"),
        ],
    }
    return render(request, "reviews/home.html", context)
