from django.db import models

class User(models.Model):
    username = models.CharField(max_length=100)
    language = models.CharField(max_length=10, default='en')

class Product(models.Model):
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=100, blank=True, null=True)
    description = models.TextField(blank=True, null=True)

class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, blank=True)
    text = models.TextField()
    language = models.CharField(max_length=10, default='en')
    created_at = models.DateTimeField(auto_now_add=True)

class Sentiment(models.Model):
    review = models.OneToOneField(Review, on_delete=models.CASCADE)
    sentiment = models.CharField(max_length=20) # positive/neutral/negative

class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
