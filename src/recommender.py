import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Map sentiment label to numeric score
SENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def build_user_item_df(reviews_df, sentiments_df):
    """
    reviews_df: DataFrame with columns ['id','user_id','product_id']
    sentiments_df: DataFrame with ['review_id','sentiment']
    returns user_item_df: rows=user_id, cols=product_id, values=avg sentiment score
    """
    # join reviews with sentiments
    df = reviews_df.merge(sentiments_df, left_on='id', right_on='review_id', how='inner')
    df['sent_score'] = df['sentiment'].map(SENT_MAP).fillna(0.0)
    agg = df.groupby(['user_id','product_id'])['sent_score'].mean().reset_index()
    user_item = agg.pivot(index='user_id', columns='product_id', values='sent_score').fillna(0.0)
    return user_item

def compute_item_similarity(user_item_df):
    """
    user_item_df: DataFrame (users x products)
    returns item_sim (DataFrame) indexed/columns by product_id
    """
    item_matrix = user_item_df.T.values  # products x users
    sim = cosine_similarity(item_matrix)
    product_ids = user_item_df.columns
    item_sim = pd.DataFrame(sim, index=product_ids, columns=product_ids)
    return item_sim

def recommend_for_user(user_id, user_item_df, item_sim, top_k=10, exclude_seen=True):
    """
    Score candidate items for user_id using item-based CF:
      score(item) = sum_over_seen_items(sim(item, seen_item) * rating_seen_item)
    """
    if user_id not in user_item_df.index:
        return []  # cold start handled outside
    user_ratings = user_item_df.loc[user_id]
    seen = user_ratings[user_ratings != 0.0].index.tolist()
    if len(seen) == 0:
        return []  # no sentiment history

    # candidate items
    candidates = item_sim.index.difference(seen)
    scores = {}
    for c in candidates:
        sims = item_sim.loc[c, seen].values
        ratings = user_ratings[seen].values
        # weighted sum
        if sims.sum() == 0:
            score = 0.0
        else:
            score = np.dot(sims, ratings) / (np.abs(sims).sum() + 1e-9)
        scores[c] = score

    # sort
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:top_k]]

# --- helper to run from Django management command or script ---
def generate_all_recommendations(reviews_df, sentiments_df, top_k=10):
    """
    Returns dict: {user_id: [product_id,..]}
    """
    user_item = build_user_item_df(reviews_df, sentiments_df)
    if user_item.shape[1] < 2:
        return {}
    item_sim = compute_item_similarity(user_item)
    recs = {}
    for user_id in user_item.index:
        recs[user_id] = recommend_for_user(user_id, user_item, item_sim, top_k=top_k)
    return recs
