import os
import pandas as pd
import tempfile
from pathlib import Path
import csv
import json
import shutil

# Import functions from your preprocessing module.
# This assumes you placed the combined script at src/preprocess_csvs.py
from src.preprocessing import clean_text, map_score_to_sentiment, preprocess_file


def test_clean_text_basic():
    s = "Hello\\nWorld   \n\n"
    out = clean_text(s)
    assert "Hello" in out and "World" in out
    assert "\n" not in out
    assert out == "Hello World"


def test_map_score_to_sentiment():
    assert map_score_to_sentiment(1) == "negative"
    assert map_score_to_sentiment(2) == "negative"
    assert map_score_to_sentiment(3) == "neutral"
    assert map_score_to_sentiment(4) == "positive"
    assert map_score_to_sentiment(5) == "positive"


def _write_small_csv(path, rows):
    # The code expects CSV with three columns: score, title, text (no header)
    with open(path, "w", newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        for score, title, text in rows:
            writer.writerow([score, title, text])


def test_preprocess_file_writes_output(tmp_path):
    # Prepare tiny input file
    inp = tmp_path / "small_train.csv"
    rows = [
        (5, "Great", "I love it"),
        (1, "Bad", "Horrible product"),
        (3, "", "It is okay"),
    ]
    _write_small_csv(inp, rows)

    out_dir = tmp_path / "processed"
    out_file = out_dir / "small_train_clean.csv"

    # Run preprocess_file; limit iterations so it stops quickly
    preprocess_file(str(inp), str(out_file), iterations=10)

    assert out_file.exists(), "Processed output file was not created"

    df = pd.read_csv(out_file)
    assert set(df.columns) == {"text", "language", "score", "sentiment"}
    # Rows should match / be deduplicated
    assert len(df) == 3
    assert "I love it" in df['text'].values
    assert "negative" in df['sentiment'].values
