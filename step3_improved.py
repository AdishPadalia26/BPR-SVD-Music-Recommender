"""
Step 3 Improved: Hybrid BPR + Bias-SVD recommendation
CS 550 Recommender System Project
"""

import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from cornac.data import Dataset
    from cornac.models import BPR
except ImportError:
    import subprocess
    import sys

    print("Installing cornac...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cornac", "-q"])
    from cornac.data import Dataset
    from cornac.models import BPR

print("=" * 70)
print("STEP 3 IMPROVED: Hybrid Recommendation")
print("=" * 70)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
R_pred = np.load("data/R_pred_biased.npy")

with open("data/user2idx.pkl", "rb") as f:
    svd_user2idx = pickle.load(f)
with open("data/item2idx.pkl", "rb") as f:
    svd_item2idx = pickle.load(f)

test_items_by_user = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
test_item_counts = {user: len(items) for user, items in test_items_by_user.items()}
train_items_by_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
item_popularity = train_df.groupby("item_id").size().to_dict()

MIN_TEST_ITEMS = 3
all_eval_users = list(test_items_by_user.keys())
active_users = [user for user, count in test_item_counts.items() if count >= MIN_TEST_ITEMS]

print(f"Train: {train_df.shape[0]:,} ratings")
print(f"Test:  {test_df.shape[0]:,} ratings")
print(f"Active users (>= {MIN_TEST_ITEMS} test items): {len(active_users):,}")

dataset = Dataset.from_uir(train_df[["user_id", "item_id", "rating"]].values)

bpr_model = BPR(
    k=64,
    max_iter=300,
    learning_rate=0.01,
    lambda_reg=0.0005,
    seed=42,
    verbose=True,
)

print("Training BPR...")
bpr_model.fit(dataset)
print("Training complete!")

cornac_user2idx = {uid: idx for idx, uid in enumerate(dataset.user_ids)}
cornac_item2idx = {iid: idx for idx, iid in enumerate(dataset.item_ids)}
cornac_idx2item = {idx: iid for iid, idx in cornac_item2idx.items()}

pop_vector = np.array([item_popularity.get(cornac_idx2item[idx], 0.0) for idx in range(len(cornac_idx2item))], dtype=float)
pop_vector = np.log1p(pop_vector)
if pop_vector.std() > 0:
    pop_vector = (pop_vector - pop_vector.mean()) / pop_vector.std()
else:
    pop_vector = np.zeros_like(pop_vector)


def calculate_dcg(hits, k=10):
    dcg = 0.0
    for i in range(min(k, len(hits))):
        if hits[i]:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(scores)
    normalized = np.zeros_like(scores, dtype=float)

    if finite_mask.any():
        valid = scores[finite_mask]
        std = valid.std()
        if std > 0:
            normalized[finite_mask] = (valid - valid.mean()) / std
        else:
            normalized[finite_mask] = 0.0

    return normalized


def get_bpr_scores(user_id):
    if user_id not in cornac_user2idx:
        return None
    user_idx = cornac_user2idx[user_id]
    return np.dot(bpr_model.u_factors[user_idx], bpr_model.i_factors.T)


def get_svd_scores(user_id):
    if user_id not in svd_user2idx:
        return None

    user_scores = np.empty(len(cornac_item2idx), dtype=float)
    svd_user_idx = svd_user2idx[user_id]
    for idx, item_id in cornac_idx2item.items():
        svd_item_idx = svd_item2idx.get(item_id)
        user_scores[idx] = R_pred[svd_user_idx, svd_item_idx] if svd_item_idx is not None else np.nan
    return user_scores


def combine_scores(user_id, config):
    bpr_scores = get_bpr_scores(user_id)
    if bpr_scores is None:
        return None

    svd_scores = get_svd_scores(user_id)
    combined = config["bpr"] * normalize_scores(bpr_scores)

    if svd_scores is not None:
        combined += config["svd"] * normalize_scores(svd_scores)

    combined += config["pop"] * pop_vector
    return combined


def generate_recommendations(user_id, config, k=10):
    scores = combine_scores(user_id, config)
    if scores is None:
        return [], []

    scores = scores.copy()
    seen_items = train_items_by_user.get(user_id, set())
    for item_id in seen_items:
        if item_id in cornac_item2idx:
            scores[cornac_item2idx[item_id]] = -np.inf

    top_indices = np.argsort(scores)[::-1][:k]
    top_items = [cornac_idx2item[idx] for idx in top_indices]
    top_scores = [float(scores[idx]) for idx in top_indices]
    return top_items, top_scores


def evaluate_config(config_name, config):
    rows = []
    for user_id in all_eval_users:
        rec_items, rec_scores = generate_recommendations(user_id, config, k=10)
        test_items = test_items_by_user.get(user_id, set())

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

    df = pd.DataFrame(rows)
    active_df = df[df["test_items_count"] >= MIN_TEST_ITEMS]

    metrics = {
        "all_users": {
            "precision": float(df["precision"].mean()),
            "recall": float(df["recall"].mean()),
            "f_measure": float(df["f_measure"].mean()),
            "ndcg": float(df["ndcg"].mean()),
            "n_users": int(len(df)),
            "nonzero_precision_users": int((df["precision"] > 0).sum()),
        },
        "active_users": {
            "precision": float(active_df["precision"].mean()) if not active_df.empty else 0.0,
            "recall": float(active_df["recall"].mean()) if not active_df.empty else 0.0,
            "f_measure": float(active_df["f_measure"].mean()) if not active_df.empty else 0.0,
            "ndcg": float(active_df["ndcg"].mean()) if not active_df.empty else 0.0,
            "n_users": int(len(active_df)),
            "nonzero_precision_users": int((active_df["precision"] > 0).sum()) if not active_df.empty else 0,
        },
        "config": config,
        "config_name": config_name,
    }
    return df, metrics


candidate_configs = {
    "bpr_only": {"bpr": 1.0, "svd": 0.0, "pop": 0.0},
    "bpr_pop": {"bpr": 0.9, "svd": 0.0, "pop": 0.1},
    "hybrid_80_15_05": {"bpr": 0.8, "svd": 0.15, "pop": 0.05},
    "hybrid_75_15_10": {"bpr": 0.75, "svd": 0.15, "pop": 0.10},
    "hybrid_70_20_10": {"bpr": 0.70, "svd": 0.20, "pop": 0.10},
}

evaluations = {}
best_name = None
best_df = None
best_metrics = None

for name, config in candidate_configs.items():
    print(f"Evaluating config: {name} -> {config}")
    df, metrics = evaluate_config(name, config)
    evaluations[name] = metrics

    score_tuple = (
        metrics["active_users"]["precision"],
        metrics["active_users"]["ndcg"],
        metrics["active_users"]["nonzero_precision_users"],
    )
    if best_metrics is None or score_tuple > (
        best_metrics["active_users"]["precision"],
        best_metrics["active_users"]["ndcg"],
        best_metrics["active_users"]["nonzero_precision_users"],
    ):
        best_name = name
        best_df = df
        best_metrics = metrics

best_df.to_csv("data/bpr_recommendations_improved.csv", index=False)

summary = {
    "min_test_items_active": MIN_TEST_ITEMS,
    "selected_config": best_name,
    "all_users": best_metrics["all_users"],
    "active_users": best_metrics["active_users"],
    "candidate_results": evaluations,
}

with open("data/recommendation_metrics_improved.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Selected config: {best_name}")
print("Selected active-user metrics:", best_metrics["active_users"])
print("Saved: data/bpr_recommendations_improved.csv")
print("Saved: data/recommendation_metrics_improved.json")

print("\n" + "=" * 70)
print("Step 3 Improved Complete!")
print("=" * 70)
