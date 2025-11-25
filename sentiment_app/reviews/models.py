from django.db import models


class User(models.Model):
    username = models.CharField(max_length=100)
    language = models.CharField(max_length=10, blank=True, null=True)

    def __str__(self):
        return self.username


class Product(models.Model):
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=100, blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name


class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, blank=True, null=True)
    text = models.TextField()
    language = models.CharField(max_length=10, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review {self.id} by {self.user}"


class Sentiment(models.Model):
    review = models.ForeignKey(Review, on_delete=models.CASCADE)
    sentiment = models.CharField(max_length=20)   # positive / neutral / negative

    def __str__(self):
        return f"Sentiment for Review {self.review.id}: {self.sentiment}"


class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)

    def __str__(self):
        return f"Recommendation: User {self.user_id} → Product {self.product_id}"
