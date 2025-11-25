from django import forms

class ReviewForm(forms.Form):
    product_id = forms.IntegerField()
    user_id = forms.IntegerField()
    title = forms.CharField(max_length=255, required=False)
    text = forms.CharField(widget=forms.Textarea)
