import os
import time
import pytest

try:
    from reviews.ml_predict import predict_sentiment
except Exception:
    try:
        from ml_predict import predict_sentiment
    except Exception:
        predict_sentiment = None

SAMPLE_TEXT = "This product arrived quickly and the battery life is excellent. Highly recommend!"
THRESH_NB = float(os.getenv("PERF_NB", "0.15"))
THRESH_SVM = float(os.getenv("PERF_SVM", "0.2"))
THRESH_LSTM = float(os.getenv("PERF_LSTM", "1.2"))

def skip_if_no_predict():
    if not predict_sentiment:
        pytest.skip("predict_sentiment not importable (skipping model perf tests)")

def warm_and_avg(model_name, runs=5):
    predict_sentiment(SAMPLE_TEXT, model_name=model_name)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_sentiment(SAMPLE_TEXT, model_name=model_name)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)

def test_nb_inference_time():
    skip_if_no_predict()
    avg = warm_and_avg("nb", runs=5)
    assert avg <= THRESH_NB, f"NB avg inference {avg:.3f}s > threshold {THRESH_NB}s"

def test_svm_inference_time():
    skip_if_no_predict()
    avg = warm_and_avg("svm", runs=5)
    assert avg <= THRESH_SVM, f"SVM avg inference {avg:.3f}s > threshold {THRESH_SVM}s"

def test_lstm_inference_time():
    skip_if_no_predict()
    avg = warm_and_avg("lstm", runs=3)
    assert avg <= THRESH_LSTM, f"LSTM avg inference {avg:.3f}s > threshold {THRESH_LSTM}s"
