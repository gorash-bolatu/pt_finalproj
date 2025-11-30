import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load traditional ML models
# -----------------------------
nb_model = joblib.load(r"reviews\models\naive_bayes_model.joblib")
svm_model = joblib.load(r"reviews\models\svm_model.joblib")

nb_vectorizer = joblib.load(r"reviews\models\naive_bayes_tfidf.joblib")
svm_vectorizer = joblib.load(r"reviews\models\svm_tfidf.joblib")

# -----------------------------
# Load LSTM model
# -----------------------------
lstm_model = tf.keras.models.load_model(r"reviews\models\lstm_model.h5")
with open(r"reviews\models\lstm_tokenizer.pkl", "rb") as f:
    lstm_tokenizer = joblib.load(f)

LSTM_MAXLEN = 200 # Must match what was used during training

# -----------------------------
# Prediction function
# -----------------------------
def predict_sentiment(text: str, model_name: str = "nb") -> str:
    """
    Predict sentiment using 'nb', 'svm', or 'lstm'.
    Returns 'positive' or 'negative'.
    """
    model_name = model_name.lower()
    
    if model_name == "nb":
        X = nb_vectorizer.transform([text])
        pred = nb_model.predict(X)[0]

    elif model_name == "svm":
        X = svm_vectorizer.transform([text])
        pred = svm_model.predict(X)[0]

    elif model_name == "lstm":
        seq = lstm_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=LSTM_MAXLEN)
        prob = lstm_model.predict(padded, verbose=0)[0][0]
        pred = 1 if prob > 0.5 else 0

    else:
        raise ValueError("Unknown model_name. Use 'nb', 'svm', or 'lstm'.")

    return "positive" if pred == 1 else "negative"
