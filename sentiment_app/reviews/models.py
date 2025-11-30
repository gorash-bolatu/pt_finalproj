from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=120, unique=True, blank=True)
    description = models.TextField(blank=True)

    class Meta:
        db_table = "Category"
        ordering = ["name"]

    def __str__(self):
        return self.name


class Product(models.Model):
    # simple product model
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True, blank=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, related_name="products")
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "Product"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    @property
    def short_description(self):
        if not self.description:
            return ""
        return (self.description[:120] + "...") if len(self.description) > 120 else self.description

