"""
Step 1: Loading and Preprocessing Amazon Musical Instruments Review Data
For CS 550 Recommender System Project

This script:
- Loads compressed JSON data
- Cleans and filters ratings
- Creates a random train/test split per user
- Saves processed data to CSV files
"""

import gzip
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

print("=" * 60)
print("PART A: Loading Data")
print("=" * 60)

file_path_gz = "Musical_Instruments_5.json.gz"
file_path_json = "Musical_Instruments_5.json"

reviews = []

if os.path.exists(file_path_gz):
    file_path = file_path_gz
    print(f"Loading from compressed file: {file_path}")
    opener = lambda f: gzip.open(f, "rt", encoding="utf-8")
else:
    file_path = file_path_json
    print(f"Loading from text file: {file_path}")
    opener = lambda f: open(f, "r", encoding="utf-8")

with opener(file_path) as f:
    for line in f:
        record = json.loads(line)
        reviews.append(
            {
                "user_id": record["reviewerID"],
                "item_id": record["asin"],
                "rating": record["overall"],
                "timestamp": record["unixReviewTime"],
            }
        )

df = pd.DataFrame(reviews)
print(f"Total rows loaded: {len(df):,}")

print("\n" + "=" * 60)
print("PART B: Basic Cleaning")
print("=" * 60)

df_clean = df.dropna(subset=["user_id", "item_id", "rating"])
print(f"After dropping missing values: {len(df_clean):,} rows")

df_clean = df_clean[(df_clean["rating"] >= 1) & (df_clean["rating"] <= 5)]
print(f"After keeping ratings 1-5: {len(df_clean):,} rows")

df_clean = df_clean.sort_values("timestamp", ascending=False)
df_clean = df_clean.drop_duplicates(subset=["user_id", "item_id"], keep="first")
df_clean = df_clean.reset_index(drop=True)
print(f"After removing duplicates: {len(df_clean):,} rows")

print("\n" + "=" * 60)
print("PART C: Dataset Summary")
print("=" * 60)

num_users = df_clean["user_id"].nunique()
num_items = df_clean["item_id"].nunique()
num_ratings = len(df_clean)
avg_rating = df_clean["rating"].mean()
max_possible_ratings = num_users * num_items
sparsity = (1 - (num_ratings / max_possible_ratings)) * 100

print(f"Total unique users:        {num_users:,}")
print(f"Total unique items:        {num_items:,}")
print(f"Total ratings:             {num_ratings:,}")
print(f"Average rating:            {avg_rating:.2f}")
print(f"Sparsity:                  {sparsity:.2f}%")

print("\n" + "=" * 60)
print("PART D: Train/Test Split")
print("=" * 60)

train_list = []
test_list = []

for user_id, user_data in df_clean.groupby("user_id"):
    n_reviews = len(user_data)

    if n_reviews == 1:
        train_list.extend(user_data.to_dict("records"))
        continue

    test_size = max(1, int(round(n_reviews * 0.2)))
    test_size = min(test_size, n_reviews - 1)

    train_user, test_user = train_test_split(
        user_data,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_list.extend(train_user.to_dict("records"))
    test_list.extend(test_user.to_dict("records"))

train_df = pd.DataFrame(train_list).sort_values(["user_id", "timestamp", "item_id"])
test_df = pd.DataFrame(test_list).sort_values(["user_id", "timestamp", "item_id"])

print(f"Training set size: {len(train_df):,} reviews")
print(f"Test set size:     {len(test_df):,} reviews")
print(f"Total:             {len(train_df) + len(test_df):,} reviews")

print("\n" + "=" * 60)
print("PART E: Saving Files")
print("=" * 60)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Done! Files saved to data/ folder")
print("  - data/train.csv")
print("  - data/test.csv")

print("\n" + "=" * 60)
print("Preprocessing complete!")
print("=" * 60)
