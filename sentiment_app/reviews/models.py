from django.db import models

class User(models.Model):
    username = models.CharField(max_length=100)
    language = models.CharField(max_length=10)

    class Meta:
        db_table = "User"        # ← match your MySQL table name exactly
        verbose_name = "User"
        verbose_name_plural = "Users"


class Product(models.Model):
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    description = models.TextField()

    class Meta:
        db_table = "Product"
        verbose_name = "Product"
        verbose_name_plural = "Products"


class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    text = models.TextField()
    language = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "Review"   # ← exact MySQL table name
        verbose_name = "Review"
        verbose_name_plural = "Reviews"


class Sentiment(models.Model):
    review = models.ForeignKey(Review, on_delete=models.CASCADE)
    sentiment = models.CharField(max_length=20)  # positive / neutral / negative

    class Meta:
        db_table = "Sentiment"
        verbose_name = "Sentiment"
        verbose_name_plural = "Sentiments"


class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)

    class Meta:
        db_table = "Recommendation"
        verbose_name = "Recommendation"
        verbose_name_plural = "Recommendations"
