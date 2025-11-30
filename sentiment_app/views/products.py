from django.shortcuts import render, get_object_or_404
from sentiment_app.models import Product
from sentiment_app.services.recommendations import RecommendationService


def product_list(request):
    products = Product.objects.all()
    return render(request, "products.html", {"products": products})


def product_detail(request, product_id):
    product = get_object_or_404(Product, id=product_id)

    recommendations = RecommendationService.recommend_for_product(product)

    return render(
        request,
        "product_detail.html",
        {
            "product": product,
            "recommendations": recommendations,
        },
    )
