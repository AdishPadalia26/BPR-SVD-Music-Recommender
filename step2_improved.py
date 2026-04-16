"""
Step 2 Improved: Bias-Corrected SVD for Rating Prediction
CS 550 Recommender System Project
"""

import json
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

print("=" * 70)
print("STEP 2 IMPROVED: Bias-Corrected SVD")
print("=" * 70)

print("\n" + "=" * 70)
print("PART A: Loading Data")
print("=" * 70)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print(f"Train: {train_df.shape[0]:,} ratings")
print(f"Test:  {test_df.shape[0]:,} ratings")

print("\n" + "=" * 70)
print("PART B: Computing Biases")
print("=" * 70)

global_mean = train_df["rating"].mean()
user_bias_df = train_df.groupby("user_id")["rating"].mean() - global_mean
item_bias_df = train_df.groupby("item_id")["rating"].mean() - global_mean

user_bias = user_bias_df.to_dict()
item_bias = item_bias_df.to_dict()

print(f"Global mean: {global_mean:.4f}")
print(f"User biases computed: {len(user_bias):,}")
print(f"Item biases computed: {len(item_bias):,}")

print("\n" + "=" * 70)
print("PART C: Building Bias-Corrected Matrix")
print("=" * 70)

all_users = train_df["user_id"].unique()
all_items = train_df["item_id"].unique()

user2idx = {user: idx for idx, user in enumerate(all_users)}
item2idx = {item: idx for idx, item in enumerate(all_items)}
idx2user = {idx: user for user, idx in user2idx.items()}
idx2item = {idx: item for item, idx in item2idx.items()}

n_users = len(user2idx)
n_items = len(item2idx)
print(f"Matrix dimensions: {n_users:,} users x {n_items:,} items")

user_indices = train_df["user_id"].map(user2idx).values
item_indices = train_df["item_id"].map(item2idx).values
ratings = train_df["rating"].values

residuals = np.array(
    [
        rating - (global_mean + user_bias.get(user_id, 0.0) + item_bias.get(item_id, 0.0))
        for user_id, item_id, rating in zip(
            train_df["user_id"].values,
            train_df["item_id"].values,
            ratings,
        )
    ]
)

R_residual_sparse = csr_matrix((residuals, (user_indices, item_indices)), shape=(n_users, n_items))
R_residual = R_residual_sparse.toarray()

print(f"Residuals computed: min={residuals.min():.2f}, max={residuals.max():.2f}")
print(f"Residual std: {residuals.std():.4f}")
print(f"Sparse residual matrix: {R_residual_sparse.nnz:,} non-zero entries")

print("\n" + "=" * 70)
print("PART D: SVD on Residuals")
print("=" * 70)

k = 50
print(f"Number of latent factors (k): {k}")

U, sigma, Vt = svds(R_residual.astype(float), k=k)
order = np.argsort(sigma)[::-1]
sigma = sigma[order]
U = U[:, order]
Vt = Vt[order, :]

R_bias = np.full((n_users, n_items), global_mean)
for u_idx, u_id in idx2user.items():
    R_bias[u_idx, :] += user_bias.get(u_id, 0.0)
for i_idx, i_id in idx2item.items():
    R_bias[:, i_idx] += item_bias.get(i_id, 0.0)

R_latent = U @ np.diag(sigma) @ Vt
R_pred = np.clip(R_bias + R_latent, 1.0, 5.0)

print(f"Predictions range: [{R_pred.min():.2f}, {R_pred.max():.2f}]")

print("\n" + "=" * 70)
print("PART E: Evaluating on Test Set")
print("=" * 70)

def fallback_predict(user_id, item_id):
    """Use the strongest available baseline when the latent model cannot score a pair."""
    baseline = global_mean
    if user_id in user_bias:
        baseline += user_bias[user_id]
    if item_id in item_bias:
        baseline += item_bias[item_id]
    return float(np.clip(baseline, 1.0, 5.0))


predicted = []
prediction_source = []

for user_id, item_id in zip(test_df["user_id"].values, test_df["item_id"].values):
    if user_id in user2idx and item_id in item2idx:
        prediction = float(R_pred[user2idx[user_id], item2idx[item_id]])
        source = "svd"
    else:
        prediction = fallback_predict(user_id, item_id)
        source = "fallback"

    predicted.append(prediction)
    prediction_source.append(source)

predicted = np.array(predicted)
actual = test_df["rating"].values

mae = np.mean(np.abs(predicted - actual))
rmse = np.sqrt(np.mean((predicted - actual) ** 2))
mae_baseline = np.mean(np.abs(actual - global_mean))
rmse_baseline = np.sqrt(np.mean((actual - global_mean) ** 2))

mae_improvement = (mae_baseline - mae) / mae_baseline * 100
rmse_improvement = (rmse_baseline - rmse) / rmse_baseline * 100
fallback_count = prediction_source.count("fallback")

print(f"Predictions made: {len(predicted):,}")
print(f"Fallback predictions used: {fallback_count:,}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE baseline:  {mae_baseline:.4f}")
print(f"RMSE baseline: {rmse_baseline:.4f}")
print(f"MAE improvement:  {mae_improvement:.2f}%")
print(f"RMSE improvement: {rmse_improvement:.2f}%")

print("\n" + "=" * 70)
print("PART F: Saving Results")
print("=" * 70)

np.save("data/R_pred_biased.npy", R_pred)
with open("data/user2idx.pkl", "wb") as f:
    pickle.dump(user2idx, f)
with open("data/item2idx.pkl", "wb") as f:
    pickle.dump(item2idx, f)
with open("data/user_bias.pkl", "wb") as f:
    pickle.dump(user_bias, f)
with open("data/item_bias.pkl", "wb") as f:
    pickle.dump(item_bias, f)

results_df = pd.DataFrame(
    {
        "user_id": test_df["user_id"].values,
        "item_id": test_df["item_id"].values,
        "actual_rating": actual,
        "predicted_rating": np.round(predicted, 4),
        "prediction_source": prediction_source,
    }
)
results_df.to_csv("data/rating_predictions_improved.csv", index=False)

metrics = {
    "model_mae": float(mae),
    "model_rmse": float(rmse),
    "baseline_mae": float(mae_baseline),
    "baseline_rmse": float(rmse_baseline),
    "mae_improvement_pct": float(mae_improvement),
    "rmse_improvement_pct": float(rmse_improvement),
    "test_rows": int(len(test_df)),
    "fallback_predictions": int(fallback_count),
}
with open("data/rating_metrics_improved.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved: data/rating_predictions_improved.csv ({len(results_df):,} rows)")
print("Saved: data/rating_metrics_improved.json")

print("\n" + "=" * 70)
print("Step 2 Improved Complete!")
print("=" * 70)
