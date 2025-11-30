from django.contrib import admin
from django.urls import path
from reviews import views as review_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", review_views.home, name="home"),   # <-- add this
    path("products/", review_views.product_list, name="product_list"),
    path("recommend/", review_views.recommend_view, name="recommend"),
]
