from django.contrib import admin
from django.urls import path
from reviews import views as review_views  # <-- correct import

urlpatterns = [
    path("admin/", admin.site.urls),

    # Product list
    path("products/", review_views.product_list, name="product_list"),

    # Recommendation endpoint
    path("recommend/", review_views.recommend_view, name="recommend"),
]
