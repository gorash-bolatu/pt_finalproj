import os
import random
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from db import engine, Base  # db.py must define Base + ORM models mapped to your MySQL schema

# Ensure tables exist
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

PROCESSED = os.path.join("data", "processed", "train_clean.csv")

NUM_PRODUCTS = 2000
NUM_USERS = 5000
SEED = 42
random.seed(SEED)


# Synthetic Generators
def gen_product_list(n):
    products = []
    for i in range(n):
        products.append({
            "name": f"Synth Product {i}",
            "category": "misc",
            "description": f"Auto-generated product description {i}"
        })
    return products


def gen_users(n):
    users = []
    for i in range(n):
        users.append({
            "username": f"user_{i}",
            "language": "unknown"
        })
    return users


# Seeding Logic
def seed_synthetic():
    session = Session()

    # 1. Insert Products
    print("Inserting products...")
    products = gen_product_list(NUM_PRODUCTS)

    insert_product_sql = text("""
        INSERT INTO Product (name, category, description)
        VALUES (:name, :category, :description)
    """)

    for p in products:
        session.execute(insert_product_sql, p)

    session.commit()

    # Fetch assigned AUTO_INCREMENT IDs
    product_ids = [row[0] for row in session.execute(text("SELECT id FROM Product")).fetchall()]
    print(f"Inserted {len(product_ids)} products.")

    # 2. Insert Users
    print("Inserting users...")
    users = gen_users(NUM_USERS)

    insert_user_sql = text("""
        INSERT INTO User (username, language)
        VALUES (:username, :language)
    """)

    for u in users:
        session.execute(insert_user_sql, u)

    session.commit()

    user_ids = [row[0] for row in session.execute(text("SELECT id FROM User")).fetchall()]
    print(f"Inserted {len(user_ids)} users.")

    # 3. Insert Reviews
    print("Loading processed reviews...")
    df = pd.read_csv(PROCESSED)
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    insert_review_sql = text("""
        INSERT INTO Review (user_id, product_id, title, text, language, created_at)
        VALUES (:user_id, :product_id, :title, :text, :language, NOW())
    """)

    for idx, row in df.iterrows():
        user_id = random.choice(user_ids)
        product_id = random.choice(product_ids)

        title = row["text"][:80]
        language = row.get("language", "unknown")

        session.execute(
            insert_review_sql,
            {
                "user_id": user_id,
                "product_id": product_id,
                "title": title,
                "text": row["text"],
                "language": language
            }
        )

        if (idx + 1) % 5000 == 0:
            session.commit()
            print(f"Inserted {idx + 1} reviews...")

    session.commit()
    session.close()

    print("Synthetic database seeding complete.")


if __name__ == "__main__":
    seed_synthetic()
