# reviews/views.py
from django.shortcuts import render
from .ml import analyze_sentiment


def home(request):
    result = None
    user_text = ""
    model_choice = "nb"  # модель по умолчанию

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
