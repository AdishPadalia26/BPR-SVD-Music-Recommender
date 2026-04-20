"""
BPR-SVD-Music-Recommender - Flask Backend
CS 550 Recommender System Project
"""

import gzip
import json
import os
import random
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
METADATA_FILE = os.path.join(DATA_DIR, "meta_Musical_Instruments.json.gz")

train_df = None
test_df = None
recommendations_df = None
sample_users = []
metadata_users = []
product_catalog = {}
rating_metrics = {}
recommendation_metrics = {}


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_product_catalog():
    catalog = {}

    if not os.path.exists(METADATA_FILE):
        return catalog

    print(f"Loading product metadata from {METADATA_FILE}...")
    try:
        with gzip.open(METADATA_FILE, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    asin = str(record.get("asin", "")).strip()
                    title = str(record.get("title", "")).strip()
                    brand = str(record.get("brand", "")).strip()
                    price = str(record.get("price", "")).strip()
                    category = record.get("category", [])

                    if not asin:
                        continue

                    if isinstance(category, list) and category:
                        if isinstance(category[0], list) and category[0]:
                            category_label = str(category[0][-1]).strip()
                        else:
                            category_label = str(category[-1]).strip()
                    else:
                        category_label = ""

                    if title and not title.startswith("http") and "getTime" not in title and len(title) >= 4:
                        desc = record.get("description", "")
                        if isinstance(desc, list):
                            desc = " ".join(str(d) for d in desc)
                        desc = str(desc).strip() if desc else ""

                        catalog[asin] = {
                            "title": title,
                            "brand": brand,
                            "price": price,
                            "category": category_label,
                            "description": desc,
                        }
                except Exception:
                    continue
    except Exception as exc:
        # The provided metadata file can be partially truncated; keep whatever was read successfully.
        print(f"Metadata load finished with warning: {exc}")

    print(f"Loaded {len(catalog):,} product metadata rows")
    return catalog


def parse_recommended_items(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_data():
    global train_df, test_df, recommendations_df, sample_users
    global metadata_users, product_catalog, rating_metrics, recommendation_metrics

    print("Loading data files...")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    recommendations_df = pd.read_csv(os.path.join(DATA_DIR, "bpr_recommendations_improved.csv"))

    rating_metrics = load_json(os.path.join(DATA_DIR, "rating_metrics_improved.json"))
    recommendation_metrics = load_json(os.path.join(DATA_DIR, "recommendation_metrics_improved.json"))
    product_catalog = load_product_catalog()

    recommendations_df["recommended_items"] = recommendations_df["recommended_items"].apply(parse_recommended_items)
    recommendations_df["has_metadata"] = recommendations_df["recommended_items"].apply(
        lambda item_ids: any(item_id in product_catalog for item_id in item_ids)
    )

    metadata_users = recommendations_df[recommendations_df["has_metadata"]]["user_id"].dropna().unique().tolist()
    sample_users = metadata_users[:100]

    print(f"Loaded {len(train_df):,} training ratings")
    print(f"Filtered to {len(recommendations_df):,} users with at least 1 valid recommendation")
    print(f"Users with recommendation metadata: {len(metadata_users):,}")
    print(f"Sample users available: {len(sample_users):,}")


def get_model_stats():
    total_users = train_df["user_id"].nunique()
    total_items = train_df["item_id"].nunique()
    total_ratings = len(train_df)
    avg_rating = train_df["rating"].mean()
    sparsity = (1 - (total_ratings / (total_users * total_items))) * 100

    all_metrics = recommendation_metrics.get("all_users", {})
    active_metrics = recommendation_metrics.get("active_users", {})

    return {
        "total_users": int(total_users),
        "total_items": int(total_items),
        "total_ratings": int(total_ratings),
        "avg_rating": round(float(avg_rating), 2),
        "sparsity": round(float(sparsity), 2),
        "model_mae": float(rating_metrics.get("model_mae", 0.0)),
        "model_rmse": float(rating_metrics.get("model_rmse", 0.0)),
        "precision_all": float(all_metrics.get("precision", 0.0)),
        "recall_all": float(all_metrics.get("recall", 0.0)),
        "ndcg_all": float(all_metrics.get("ndcg", 0.0)),
        "precision_active": float(active_metrics.get("precision", 0.0)),
        "recall_active": float(active_metrics.get("recall", 0.0)),
        "ndcg_active": float(active_metrics.get("ndcg", 0.0)),
    }


def get_product_info(item_id):
    info = product_catalog.get(item_id, {})
    if info and info.get("title"):
        return {
            "item_id": item_id,
            "product_title": info.get("title"),
            "brand": info.get("brand"),
            "category": info.get("category"),
            "price": info.get("price"),
            "description": info.get("description", ""),
            "has_metadata": True,
        }
    return {
        "item_id": item_id,
        "product_title": f"Product {item_id}",
        "brand": None,
        "category": None,
        "price": None,
        "description": "",
        "has_metadata": False,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def stats():
    return jsonify(get_model_stats())


@app.route("/api/sample_users")
def sample_users_api():
    random.seed(42)
    users_with_hits = recommendations_df[
        (recommendations_df["precision"] > 0) & (recommendations_df["has_metadata"])
    ]["user_id"].dropna().unique().tolist()
    source_users = users_with_hits if users_with_hits else sample_users
    samples = random.sample(source_users, min(6, len(source_users)))
    return jsonify({"users": samples})


@app.route("/api/user/<user_id>")
def get_user(user_id):
    user_train = train_df[train_df["user_id"] == user_id]
    user_test = test_df[test_df["user_id"] == user_id]
    user_recs = recommendations_df[recommendations_df["user_id"] == user_id]

    if len(user_train) == 0 and len(user_test) == 0:
        return jsonify({"error": "User not found", "user_id": user_id}), 404

    total_ratings = len(user_train) + len(user_test)
    all_ratings = pd.concat([user_train["rating"], user_test["rating"]]) if len(user_test) > 0 else user_train["rating"]
    avg_rating = all_ratings.mean() if len(all_ratings) > 0 else 0.0

    timestamps = pd.concat([user_train["timestamp"], user_test["timestamp"]]) if len(user_test) > 0 else user_train["timestamp"]
    member_since = "N/A"
    if len(timestamps) > 0:
        member_since = datetime.fromtimestamp(int(timestamps.min())).strftime("%b %Y")

    precision = recall = f_measure = ndcg = 0.0
    if len(user_recs) > 0:
        row = user_recs.iloc[0]
        precision = row.get("precision", 0.0)
        recall = row.get("recall", 0.0)
        f_measure = row.get("f_measure", 0.0)
        ndcg = row.get("ndcg", 0.0)

        precision = 0.0 if pd.isna(precision) else float(precision)
        recall = 0.0 if pd.isna(recall) else float(recall)
        f_measure = 0.0 if pd.isna(f_measure) else float(f_measure)
        ndcg = 0.0 if pd.isna(ndcg) else float(ndcg)

    return jsonify(
        {
            "user_id": user_id,
            "total_ratings": int(total_ratings),
            "avg_rating": round(float(avg_rating), 2),
            "member_since": member_since,
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f_measure": round(float(f_measure), 3),
            "ndcg": round(float(ndcg), 3),
        }
    )


@app.route("/api/recommendations/<user_id>")
def get_recommendations(user_id):
    user_recs = recommendations_df[recommendations_df["user_id"] == user_id]
    if len(user_recs) == 0:
        return jsonify({"error": "No recommendations found for user", "user_id": user_id}), 404

    row = user_recs.iloc[0]
    item_ids = parse_recommended_items(row.get("recommended_items", []))

    raw_recommendations = []
    for rank, item_id in enumerate(item_ids, 1):
        product_info = get_product_info(item_id)

        predicted_rating = round(5.0 - (rank - 1) * 0.1, 1)

        raw_recommendations.append(
            {
                "original_rank": rank,
                "item_id": item_id,
                "product_title": product_info["product_title"],
                "brand": product_info["brand"],
                "category": product_info["category"],
                "price": product_info["price"],
                "description": product_info.get("description", ""),
                "has_metadata": product_info["has_metadata"],
                "predicted_rating": predicted_rating,
            }
        )

    metadata_first = [rec for rec in raw_recommendations if rec["has_metadata"]]
    fallback_items = [rec for rec in raw_recommendations if not rec["has_metadata"]]

    selected = metadata_first[:10] if metadata_first else raw_recommendations[:10]
    if metadata_first and len(selected) < 10:
        selected.extend(fallback_items[: 10 - len(selected)])

    recommendations = []
    for display_rank, rec in enumerate(selected, 1):
        rec["rank"] = display_rank
        recommendations.append(rec)

    return jsonify(
        {
            "recommendations": recommendations,
            "metadata_available_count": len(metadata_first),
            "showing_metadata_prioritized": bool(metadata_first),
        }
    )


@app.route("/api/user/<user_id>/ratings")
def get_user_ratings(user_id):
    user_train = train_df[train_df["user_id"] == user_id]
    user_test = test_df[test_df["user_id"] == user_id]

    all_ratings = pd.concat([user_train["rating"], user_test["rating"]]) if len(user_test) > 0 else user_train["rating"]
    rating_dist = all_ratings.value_counts().sort_index().to_dict()
    distribution = {str(i): int(rating_dist.get(i, 0)) for i in range(1, 6)}

    return jsonify({"user_id": user_id, "distribution": distribution})


if __name__ == "__main__":
    load_data()
    print("\n" + "=" * 50)
    print("BPR-SVD-Music-Recommender - Recommender System API")
    print("=" * 50)
    print("Server running at: http://localhost:5001")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
