# reviews/ml.py

from pathlib import Path
import joblib
import numpy as np

# пробуем аккуратно подключить keras (для LSTM)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pickle
    TENSORFLOW_AVAILABLE = True
except ImportError:
    load_model = None
    pad_sequences = None
    pickle = None
    TENSORFLOW_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent  # папка reviews


# ============ 1. Naive Bayes ============

NB_MODEL_PATH = BASE_DIR / "naive_bayes_model.joblib"
NB_VECTORIZER_PATH = BASE_DIR / "naive_bayes_tfidf.joblib"
NB_ENCODER_PATH = BASE_DIR / "naive_bayes_label_encoder.joblib"

try:
    nb_model = joblib.load(NB_MODEL_PATH)
    nb_vectorizer = joblib.load(NB_VECTORIZER_PATH)
    nb_encoder = joblib.load(NB_ENCODER_PATH)
    print("✔ Naive Bayes loaded")
except Exception as e:
    print("❌ Error loading Naive Bayes:", e)
    nb_model = nb_vectorizer = nb_encoder = None


# ============ 2. SVM ============

SVM_MODEL_PATH = BASE_DIR / "svm_model.joblib"
SVM_VECTORIZER_PATH = BASE_DIR / "svm_tfidf.joblib"
SVM_ENCODER_PATH = BASE_DIR / "svm_label_encoder.joblib"

try:
    svm_model = joblib.load(SVM_MODEL_PATH)
    svm_vectorizer = joblib.load(SVM_VECTORIZER_PATH)
    svm_encoder = joblib.load(SVM_ENCODER_PATH)
    print("✔ SVM loaded")
except Exception as e:
    print("❌ Error loading SVM:", e)
    svm_model = svm_vectorizer = svm_encoder = None


# ============ 3. LSTM ============

LSTM_MODEL_PATH = BASE_DIR / "lstm_model.h5"
LSTM_TOKENIZER_PATH = BASE_DIR / "lstm_tokenizer.pkl"
LSTM_ENCODER_PATH = BASE_DIR / "lstm_label_encoder.joblib"

if TENSORFLOW_AVAILABLE:
    try:
        lstm_model = load_model(LSTM_MODEL_PATH)
        with open(LSTM_TOKENIZER_PATH, "rb") as f:
            lstm_tokenizer = pickle.load(f)
        lstm_encoder = joblib.load(LSTM_ENCODER_PATH)
        MAX_LEN = 100  # ту длину, которую ты использовала при обучении
        print("✔ LSTM loaded")
    except Exception as e:
        print("❌ Error loading LSTM:", e)
        lstm_model = lstm_tokenizer = lstm_encoder = None
        MAX_LEN = 100
else:
    lstm_model = lstm_tokenizer = lstm_encoder = None
    MAX_LEN = 100


def analyze_sentiment(text: str, model_type: str = "nb") -> str:
    """
    model_type: 'nb' | 'svm' | 'lstm'
    Возвращает строку: 'Positive' / 'Negative' / 'Neutral' (или метку).
    """
    text = text.strip()
    if not text:
        return "Neutral"

    # --- Naive Bayes ---
    if model_type == "nb" and nb_model and nb_vectorizer and nb_encoder:
        X = nb_vectorizer.transform([text])
        pred = nb_model.predict(X)[0]
        label = nb_encoder.inverse_transform([pred])[0]
        return str(label).capitalize()

    # --- SVM ---
    if model_type == "svm" and svm_model and svm_vectorizer and svm_encoder:
        X = svm_vectorizer.transform([text])
        pred = svm_model.predict(X)[0]
        label = svm_encoder.inverse_transform([pred])[0]
        return str(label).capitalize()

    # --- LSTM ---
    if model_type == "lstm" and lstm_model and lstm_tokenizer and lstm_encoder:
        seq = lstm_tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=MAX_LEN)
        probs = lstm_model.predict(seq)[0]
        pred_class = int(np.argmax(probs))
        label = lstm_encoder.inverse_transform([pred_class])[0]
        return str(label).capitalize()

    return "Error"
