"""
Step 3: Top-10 Item Recommendation & Evaluation
For CS 550 Recommender System Project

This script:
- Generates top-10 item recommendations for each user
- Evaluates recommendations using Precision@10, Recall@10, F-measure, NDCG@10
"""

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm  # Progress bar

# ============================================================
# PART A — LOAD EVERYTHING
# ============================================================
print("=" * 60)
print("PART A: Loading Data")
print("=" * 60)

# Load train and test CSV files
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Load the predicted rating matrix (from Step 2)
R_pred = np.load('data/R_pred.npy')

# Load user and item mappings
with open('data/user2idx.pkl', 'rb') as f:
    user2idx = pickle.load(f)

with open('data/item2idx.pkl', 'rb') as f:
    item2idx = pickle.load(f)

# Create reverse mapping: index → item_id (for recommendations)
idx2item = {idx: item for item, idx in item2idx.items()}

# Print confirmation of what we loaded
print(f"Train data:        {train_df.shape}")
print(f"Test data:         {test_df.shape}")
print(f"R_pred matrix:     {R_pred.shape}")
print(f"User mappings:     {len(user2idx):,} users")
print(f"Item mappings:     {len(item2idx):,} items")

# ============================================================
# PART B — BUILD PER-USER LOOKUPS
# ============================================================
print("\n" + "=" * 60)
print("PART B: Building Per-User Lookups")
print("=" * 60)

# For each user, track which items they've already rated
# These are items we should NOT recommend (already seen)

# Dictionary: user_id -> set of item_ids they rated in training
train_items_by_user = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
print(f"Users with training data: {len(train_items_by_user):,}")

# Dictionary: user_id -> set of item_ids they rated in test (ground truth)
test_items_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
print(f"Users with test data:     {len(test_items_by_user):,}")

# Calculate average number of test items per user
avg_test_items = np.mean([len(items) for items in test_items_by_user.values()])
print(f"Average test items per user: {avg_test_items:.2f}")

# ============================================================
# PART C — GENERATE TOP-10 RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 60)
print("PART C: Generating Top-10 Recommendations")
print("=" * 60)

# Get list of users who have test data (we'll recommend for them)
users_to_recommend = list(test_items_by_user.keys())
print(f"Generating recommendations for {len(users_to_recommend):,} users...")

# Dictionary to store recommendations: user_id -> [item1, item2, ..., item10]
recommendations = {}

# Create numpy array for faster sorting
# Pre-convert user IDs to indices
user_indices = [user2idx[uid] for uid in users_to_recommend if uid in user2idx]
valid_users = [uid for uid in users_to_recommend if uid in user2idx]

print(f"Valid users (in training): {len(valid_users):,}")
print("Generating recommendations (with progress bar)...")

# Process users with progress bar
for i, user_id in enumerate(tqdm(valid_users, desc="Recommending")):
    # Get the user's row index in R_pred
    user_idx = user2idx[user_id]
    
    # Get predicted ratings for this user
    user_predictions = R_pred[user_idx].copy()
    
    # Get items this user has already rated in training
    seen_items = train_items_by_user.get(user_id, set())
    
    # Convert seen item IDs to indices and zero out their predictions
    for item_id in seen_items:
        if item_id in item2idx:
            item_idx = item2idx[item_id]
            user_predictions[item_idx] = -np.inf  # Exclude from recommendations
    
    # Sort by predicted rating (descending) and get top 10 item indices
    top_10_indices = np.argsort(user_predictions)[::-1][:10]
    
    # Convert item indices back to item IDs
    top_10_items = [idx2item[idx] for idx in top_10_indices]
    
    # Store recommendations
    recommendations[user_id] = top_10_items
    
    # Print progress every 5000 users
    if (i + 1) % 5000 == 0:
        print(f"Processed {i + 1:,} / {len(valid_users):,} users")

print(f"Generated recommendations for {len(recommendations):,} users")

# ============================================================
# PART D — EVALUATE RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 60)
print("PART D: Evaluating Recommendations")
print("=" * 60)

def calculate_dcg(hit_list, k=10):
    """
    Calculate Discounted Cumulative Gain (DCG)
    DCG = sum of (rel_i / log2(i+1)) for i = 0 to k-1
    For recommendations: rel_i = 1 if item is hit, 0 otherwise
    """
    dcg = 0.0
    for i, is_hit in enumerate(hit_list[:k]):
        if is_hit:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
    return dcg

def calculate_ndcg(recommended_items, test_items, k=10):
    """
    Calculate Normalized DCG
    NDCG = DCG / IDCG (where IDCG is ideal DCG with all hits at top)
    """
    # Check which recommended items are in test set
    hit_list = [1 if item in test_items else 0 for item in recommended_items[:k]]
    
    # Calculate DCG
    dcg = calculate_dcg(hit_list, k)
    
    # Calculate ideal DCG (all hits sorted by position)
    ideal_hit_list = [1] * min(len(test_items), k)
    idcg = calculate_dcg(ideal_hit_list, k)
    
    # Return NDCG (1.0 if perfect)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# Lists to store metrics for each user
precision_list = []
recall_list = []
f_measure_list = []
ndcg_list = []

print("Calculating metrics for each user...")

for user_id in tqdm(recommendations.keys(), desc="Evaluating"):
    # Get this user's recommendations and test items
    rec_items = recommendations[user_id]
    test_items = test_items_by_user.get(user_id, set())
    
    if len(test_items) == 0:
        continue  # Skip users with no test items
    
    # Count how many recommended items are in the test set (hits)
    hits = sum(1 for item in rec_items if item in test_items)
    
    # Precision@10 = hits / 10
    precision = hits / 10.0
    precision_list.append(precision)
    
    # Recall@10 = hits / |test items|
    recall = hits / len(test_items)
    recall_list.append(recall)
    
    # F-measure = 2 * (Precision * Recall) / (Precision + Recall)
    if precision + recall > 0:
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0.0
    f_measure_list.append(f_measure)
    
    # NDCG@10
    ndcg = calculate_ndcg(rec_items, test_items, k=10)
    ndcg_list.append(ndcg)

# Calculate averages
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f_measure = np.mean(f_measure_list)
avg_ndcg = np.mean(ndcg_list)

# Print results
print("\n" + "=" * 60)
print("EVALUATION RESULTS (Averaged over all users)")
print("=" * 60)
print(f"Precision@10:      {avg_precision:.4f}")
print(f"Recall@10:        {avg_recall:.4f}")
print(f"F-measure@10:     {avg_f_measure:.4f}")
print(f"NDCG@10:          {avg_ndcg:.4f}")

print("\nExplanation:")
print("- Precision@10: Of the 10 recommended items, what fraction was relevant?")
print("- Recall@10: Of all relevant items, how many did we recommend?")
print("- F-measure: Harmonic mean of Precision and Recall")
print("- NDCG@10: How close our ranking is to the ideal ranking")

# ============================================================
# PART E — SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("PART E: Saving Results")
print("=" * 60)

# Build results dataframe
results_data = []
for user_id in recommendations.keys():
    rec_items = recommendations[user_id]
    test_items = test_items_by_user.get(user_id, set())
    
    hits = sum(1 for item in rec_items if item in test_items)
    precision = hits / 10.0
    recall = hits / len(test_items) if len(test_items) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    ndcg = calculate_ndcg(rec_items, test_items, k=10)
    
    results_data.append({
        'user_id': user_id,
        'recommended_items': ','.join(rec_items),
        'num_hits': hits,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f_measure': round(f_measure, 4),
        'ndcg': round(ndcg, 4)
    })

results_df = pd.DataFrame(results_data)
results_df.to_csv('data/recommendations.csv', index=False)
print(f"Saved: data/recommendations.csv ({len(results_df):,} rows)")

# Print summary table
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"{'Metric':<20} {'Value':>10}")
print("-" * 32)
print(f"{'Precision@10':<20} {avg_precision:>10.4f}")
print(f"{'Recall@10':<20} {avg_recall:>10.4f}")
print(f"{'F-measure@10':<20} {avg_f_measure:>10.4f}")
print(f"{'NDCG@10':<20} {avg_ndcg:>10.4f}")
print("=" * 60)
print("Step 3 Complete!")
print("=" * 60)
