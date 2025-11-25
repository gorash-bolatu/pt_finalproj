from django.contrib import admin
from .models import User, Product, Review, Sentiment, Recommendation

admin.site.register(User)
admin.site.register(Product)
admin.site.register(Review)
admin.site.register(Sentiment)
admin.site.register(Recommendation)
