"""
Step 3 Baseline: Top-10 Recommendation using Bias-SVD predictions
CS 550 Recommender System Project
"""

import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 70)
print("STEP 3 BASELINE: SVD Top-10 Recommendation")
print("=" * 70)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
R_pred = np.load("data/R_pred_biased.npy")

with open("data/user2idx.pkl", "rb") as f:
    user2idx = pickle.load(f)
with open("data/item2idx.pkl", "rb") as f:
    item2idx = pickle.load(f)

idx2item = {idx: item for item, idx in item2idx.items()}
train_items_by_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
test_items_by_user = test_df.groupby("user_id")["item_id"].apply(set).to_dict()

MIN_TEST_ITEMS = 3


def calculate_dcg(hits, k=10):
    dcg = 0.0
    for i in range(min(k, len(hits))):
        if hits[i]:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def generate_recommendations(user_id, k=10):
    if user_id not in user2idx:
        return [], []

    user_scores = R_pred[user2idx[user_id]].copy()
    for item_id in train_items_by_user.get(user_id, set()):
        item_idx = item2idx.get(item_id)
        if item_idx is not None:
            user_scores[item_idx] = -np.inf

    top_indices = np.argsort(user_scores)[::-1][:k]
    top_items = [idx2item[idx] for idx in top_indices]
    top_scores = [float(user_scores[idx]) for idx in top_indices]
    return top_items, top_scores


rows = []
for user_id in tqdm(test_items_by_user.keys(), desc="Evaluating SVD baseline"):
    rec_items, rec_scores = generate_recommendations(user_id, k=10)
    test_items = test_items_by_user[user_id]

    hit_list = [1 if item in test_items else 0 for item in rec_items]
    hits = sum(hit_list)
    precision = hits / 10.0 if rec_items else 0.0
    recall = hits / len(test_items) if test_items else 0.0
    f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    idcg = calculate_dcg([1] * min(10, len(test_items)))
    ndcg = calculate_dcg(hit_list) / idcg if idcg > 0 else 0.0

    rows.append(
        {
            "user_id": user_id,
            "recommended_items": ",".join(rec_items),
            "recommended_scores": ",".join(f"{score:.6f}" for score in rec_scores),
            "num_hits": hits,
            "test_items_count": len(test_items),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f_measure": round(f_measure, 4),
            "ndcg": round(ndcg, 4),
        }
    )

results_df = pd.DataFrame(rows)
results_df.to_csv("data/svd_recommendations_improved.csv", index=False)

all_metrics = {
    "precision": float(results_df["precision"].mean()),
    "recall": float(results_df["recall"].mean()),
    "f_measure": float(results_df["f_measure"].mean()),
    "ndcg": float(results_df["ndcg"].mean()),
    "n_users": int(len(results_df)),
}

active_df = results_df[results_df["test_items_count"] >= MIN_TEST_ITEMS]
active_metrics = {
    "precision": float(active_df["precision"].mean()) if not active_df.empty else 0.0,
    "recall": float(active_df["recall"].mean()) if not active_df.empty else 0.0,
    "f_measure": float(active_df["f_measure"].mean()) if not active_df.empty else 0.0,
    "ndcg": float(active_df["ndcg"].mean()) if not active_df.empty else 0.0,
    "n_users": int(len(active_df)),
}

summary = {
    "min_test_items_active": MIN_TEST_ITEMS,
    "all_users": all_metrics,
    "active_users": active_metrics,
}
with open("data/svd_recommendation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved: data/svd_recommendations_improved.csv ({len(results_df):,} rows)")
print("Saved: data/svd_recommendation_metrics.json")
print("All-user metrics:", all_metrics)
print("Active-user metrics:", active_metrics)

print("\n" + "=" * 70)
print("Step 3 SVD Baseline Complete!")
print("=" * 70)
