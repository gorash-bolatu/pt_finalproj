import joblib
import pickle
import tensorflow as tf
from django.core.management.base import BaseCommand
from reviews.models import Review, Sentiment, Recommendation, Product, User
from django.db import transaction
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = "models"

class Command(BaseCommand):
    help = "Run sentiment inference on unlabeled reviews and populate recommendations."

    def handle(self, *args, **options):
        # load artifacts
        tfidf = joblib.load(f"{MODEL_DIR}/svm_tfidf.joblib")
        svm = joblib.load(f"{MODEL_DIR}/svm_model.joblib")
        label_enc = joblib.load(f"{MODEL_DIR}/svm_label_encoder.joblib")

        # LSTM artifacts (if needed)
        tokenizer = pickle.load(open(f"{MODEL_DIR}/lstm_tokenizer.pkl","rb"))
        lstm = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_model.h5")

        # process unlabeled reviews (Sentiment table absent or review has no sentiment)
        qs = Review.objects.filter(sentiment__isnull=True)  # if using Review.sentiment
        batch = 500
        count = qs.count()
        self.stdout.write(f"Found {count} unlabeled reviews")
        idx = 0
        while True:
            chunk = qs[idx: idx + batch]
            if not chunk:
                break
            texts = [r.text for r in chunk]
            X = tfidf.transform(texts)
            preds = svm.predict(X)
            # convert preds to str label if you used LabelEncoder
            labels = label_enc.inverse_transform(preds)
            with transaction.atomic():
                for r, lab in zip(chunk, labels):
                    # create or update Sentiment row (or update Review.sentiment)
                    Sentiment.objects.update_or_create(review=r, defaults={"sentiment": lab})
            idx += batch
            self.stdout.write(f"Processed {idx}/{count}")
        # generate recommendations (simple popularity-based)
        self.stdout.write("Generating popularity recommendations")
        # get product ids with most positive sentiments
        from django.db.models import Count
        top_products = Sentiment.objects.filter(sentiment="positive").values('review__product').annotate(cnt=Count('id')).order_by('-cnt')[:20]
        top_product_ids = [tp['review__product'] for tp in top_products]
        # For each user, recommend top-k excluding products they already reviewed
        users = User.objects.all()
        for u in users:
            reviewed = set(Review.objects.filter(user=u).values_list('product', flat=True))
            recs = [pid for pid in top_product_ids if pid not in reviewed][:5]
            with transaction.atomic():
                # remove old recs
                Recommendation.objects.filter(user=u).delete()
                for pid in recs:
                    Recommendation.objects.create(user=u, product=Product.objects.get(pk=pid))
        self.stdout.write("Inference and recommendation done.")
