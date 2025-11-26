import pandas as pd
import os
import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

CHUNKSIZE = 300  # adjust based on RAM

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\\n", " ").replace("\\"+"n", " ")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def map_score_to_sentiment(score):
    score = int(score)
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

def preprocess_file(input_path, output_path, iterations=None):
    if os.path.exists(output_path):
        raise FileExistsError()
    print("Processing", output_path + ":")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    first = True
    for i, chunk in enumerate(pd.read_csv(input_path, header=None, names=['score','title','text'],
                                         chunksize=CHUNKSIZE, quoting=1, engine='python')):
        # combine title+text
        chunk['text'] = (chunk['title'].fillna('') + ". " + chunk['text'].fillna('')).apply(clean_text)
        chunk = chunk[['score','text']]
        # deduplicate
        chunk = chunk.drop_duplicates(subset=['text'])
        # detect language
        langs = []
        for txt in chunk['text'].tolist():
            try:
                langs.append(detect(txt) if txt.strip() else 'unknown')
            except Exception:
                langs.append('unknown')
        chunk['language'] = langs
        # map sentiment
        chunk['sentiment'] = chunk['score'].apply(map_score_to_sentiment)
        chunk = chunk[['text','language','score','sentiment']]
        # write
        chunk.to_csv(output_path, index=False, mode='a', header=first)
        first = False
        print(f"\tProcessed chunk {i+1}" + (f"/{iterations}" if iterations else ""), f"({(i+1)*CHUNKSIZE} rows)", end='\r')
        if iterations and (i+1 >= iterations):
            break
    print(f"\nFinished processing {input_path}")

if __name__ == "__main__":
    files_to_process = [
        {"input": r"..\data\raw\train.csv", "output": r"..\data\processed\train_clean.csv", "iterations": 80},
        {"input": r"..\data\raw\test.csv",  "output": r"..\data\processed\test_clean.csv",  "iterations": 20},
    ]
    for file in files_to_process:
        preprocess_file(file['input'], file['output'], file['iterations'])
