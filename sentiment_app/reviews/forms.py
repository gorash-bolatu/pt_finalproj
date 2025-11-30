from django import forms

MODEL_CHOICES = [
    ("nb", "Naive Bayes"),
    ("svm", "SVM"),
    ("lstm", "LSTM"),
]

class ReviewForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 4, "cols": 40}),
        label="Your review",
        required=True,
    )
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="ML Model",
        required=False,
        initial="nb",
    )
