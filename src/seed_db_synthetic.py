import os
import random
import hashlib
import pandas as pd
from sqlalchemy.orm import sessionmaker
from db import engine, Base  # db.py defines Product, Review, User via ORM

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

PROCESSED = os.path.join("data", "processed", "train_clean.csv")  # or test_clean.csv
NUM_PRODUCTS = 2000
NUM_USERS = 5000
SEED = 42
random.seed(SEED)

def gen_product_list(n):
    # create simple product entries
    products = []
    for i in range(n):
        pid = f"P{100000+i}"
        products.append({"id": pid, "title": f"Synth Product {i}", "category": "misc"})
    return products

def gen_users(n):
    users = []
    for i in range(n):
        uid = f"U{10000+i}"
        users.append({"username": uid, "language": "unknown"})
    return users

def seed_synthetic():
    session = Session()
    products = gen_product_list(NUM_PRODUCTS)
    users = gen_users(NUM_USERS)

    # Bulk insert via raw SQL for speed (or use ORM)
    for p in products:
        session.execute(
            "INSERT IGNORE INTO Product (id, title, category, price) VALUES (%s,%s,%s,%s)",
            (p['id'], p['title'], p['category'], None)
        )
    for u in users:
        session.execute(
            "INSERT IGNORE INTO User (username, language) VALUES (%s,%s)",
            (u['username'], u['language'])
        )
    session.commit()

    # load processed reviews and assign random user/product
    df = pd.read_csv(PROCESSED)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    for idx, row in df.iterrows():
        product_choice = random.choice(products)['id']
        user_choice = random.choice(users)['username']
        title = (row.get('text')[:80])  # use first part as title
        text = row['text']
        # rating -> use score if exists else map from sentiment
        rating = float(row.get('score') if 'score' in row and not pd.isna(row.get('score')) else (5 if row['sentiment']=='positive' else 1))
        session.execute(
            "INSERT INTO Review (user_id, product_id, title, text, language, created_at) VALUES (%s,%s,%s,%s,%s,NOW())",
            (user_choice, product_choice, title, text, row.get('language','unknown'))
        )
        if (idx+1) % 10000 == 0:
            session.commit()
            print("Inserted", idx+1, "reviews")
    session.commit()
    session.close()
    print("Synthetic seeding done.")

if __name__ == "__main__":
    seed_synthetic()
