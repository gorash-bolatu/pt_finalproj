import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# Config 
DATA_PATH = os.path.join("data", "processed", "train_clean.csv")
OUT_DIR = os.path.join("models")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MAX_SAMPLES = 20000    # set lower for quick runs; set to None to use all
TEST_SIZE = 0.1

# LSTM params
MAX_VOCAB = 30000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 5   # increase if you have time/GPU

MODEL_NAME = "lstm"

# Load data 
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'sentiment'])
if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)

texts = df['text'].astype(str).values
labels = df['sentiment'].astype(str).values

# Encode labels 
le = LabelEncoder()
y_enc = le.fit_transform(labels)
num_classes = len(le.classes_)

# Train/Val split 
X_train, X_val, y_train, y_val = train_test_split(
    texts, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)

# Tokenize 
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
Xtr_seq = tokenizer.texts_to_sequences(X_train)
Xv_seq = tokenizer.texts_to_sequences(X_val)

Xtr = pad_sequences(Xtr_seq, maxlen=MAX_LEN, padding='post', truncating='post')
Xv = pad_sequences(Xv_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Build model 
tf.keras.backend.clear_session()
model = models.Sequential([
    layers.Embedding(input_dim=min(MAX_VOCAB, len(tokenizer.word_index) + 1),
                     output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train 
history = model.fit(
    Xtr, y_train,
    validation_data=(Xv, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Eval 
y_prob = model.predict(Xv, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_prob, axis=1)
report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)

# Save artifacts 
# Keras model
model_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_model.h5")
model.save(model_path)

# tokenizer
tokenizer_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_tokenizer.pkl")
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

# label encoder
le_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_label_encoder.joblib")
import joblib
joblib.dump(le, le_path)

# configs and metrics
config = {
    "model": MODEL_NAME,
    "max_vocab": MAX_VOCAB,
    "max_len": MAX_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "lstm_units": LSTM_UNITS,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "random_state": RANDOM_STATE,
    "max_samples": MAX_SAMPLES,
    "test_size": TEST_SIZE
}
with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_config.json"), "w", encoding="utf8") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_metrics.json"), "w", encoding="utf8") as f:
    json.dump(report, f, indent=2)

print("LSTM training complete. Artifacts saved to", OUT_DIR)
print("Sample evaluation (macro avg):", report.get("macro avg", {}))
