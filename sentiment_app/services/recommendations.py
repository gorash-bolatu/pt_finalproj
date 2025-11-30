import random

class RecommendationService:
    @staticmethod
    def recommend_for_product(product):
        # Placeholder logic — replace with ML later
        related = product.category.products.exclude(id=product.id)[:10]
        if not related:
            return []
        return random.sample(list(related), min(3, len(related)))
