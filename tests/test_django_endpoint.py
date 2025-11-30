from django.test import TestCase, Client
from django.urls import reverse
from reviews.models import User, Product, Review, Sentiment, Recommendation

class ReviewEndpointIntegrationTests(TestCase):
    # This test suite uses the Django testing DB; no external DB is touched.
    def setUp(self):
        self.client = Client()
        # create test user and products
        self.user = User.objects.create(username="testuser", language="en")
        self.prod1 = Product.objects.create(id="P1", name="Product 1", category="cat")
        self.prod2 = Product.objects.create(id="P2", name="Product 2", category="cat")
        # create a review (unscored)
        self.review = Review.objects.create(user=self.user, product=self.prod1, title="T", text="Nice!", language="en")

    def test_submit_review_endpoint(self):
        """
        Assumes you have a POST endpoint at /reviews/submit/ that accepts
        payload: {"username": "...", "product_id": "...", "title": "...", "text": "...", "language": "..."}
        and returns 201 on success.
        """
        payload = {
            "username": "newuser",
            "product_id": "P3",
            "title": "Hello",
            "text": "I like this product",
            "language": "en"
        }
        resp = self.client.post("/reviews/submit/", data=payload, content_type="application/json")
        assert resp.status_code in (200, 201), f"Unexpected status: {resp.status_code}, body: {resp.content}"

    def test_get_recommendations_for_user(self):
        """
        Assumes you have a recommendations endpoint at /reviews/<user_id>/recommendations/
        that returns JSON like {"recommended_products": ["P1","P2"], "user_sentiment": "positive"}.
        """
        url = f"/reviews/{self.user.id}/recommendations/"
        resp = self.client.get(url)
        assert resp.status_code in (200, 204), f"Unexpected status: {resp.status_code}"
        # If 200, expect JSON structure
        if resp.status_code == 200:
            data = resp.json()
            assert "recommended_products" in data
            assert isinstance(data["recommended_products"], list)
