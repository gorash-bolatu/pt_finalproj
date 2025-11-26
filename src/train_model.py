import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(train_path='data/processed/train.csv', val_path='data/processed/val.csv'):
    train = pd.read_csv(train_path)   # expect columns: text,sentiment
    val = pd.read_csv(val_path)
    return train, val

def train_and_save(train, val):
    X_train = train['text'].fillna("")
    y_train = train['sentiment']
    X_val = val['text'].fillna("")
    y_val = val['sentiment']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=5)),
        ('clf', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000))
    ])

    param_grid = {
        'clf__C': [0.1, 1.0, 5.0]
    }

    gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    best = gs.best_estimator_

    # Eval
    preds = best.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, preds))

    # Save model
    joblib.dump(best, os.path.join(MODEL_DIR, 'logreg_tfidf.pkl'))
    print("Saved model to", os.path.join(MODEL_DIR, 'logreg_tfidf.pkl'))

if __name__ == "__main__":
    train, val = load_data()
    train_and_save(train, val)
