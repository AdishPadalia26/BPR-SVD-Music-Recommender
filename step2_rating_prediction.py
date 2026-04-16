"""
Step 2: Rating Prediction using Matrix Factorization (SVD)
For CS 550 Recommender System Project

Uses sparse SVD for better performance on large matrices
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pickle

print("=" * 60)
print("PART A: Loading Train and Test Data")
print("=" * 60)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train data shape:  {train_df.shape}")
print(f"Test data shape:   {test_df.shape}")

print("\n" + "=" * 60)
print("PART B: Building User-Item Matrix")
print("=" * 60)

all_users = train_df['user_id'].unique()
all_items = train_df['item_id'].unique()

user2idx = {user: idx for idx, user in enumerate(all_users)}
item2idx = {item: idx for idx, item in enumerate(all_items)}

n_users = len(user2idx)
n_items = len(item2idx)

print(f"Number of unique users: {n_users:,}")
print(f"Number of unique items: {n_items:,}")

user_indices = train_df['user_id'].map(user2idx).values
item_indices = train_df['item_id'].map(item2idx).values
ratings = train_df['rating'].values

R_sparse = csr_matrix(
    (ratings, (user_indices, item_indices)),
    shape=(n_users, n_items)
)

global_mean = train_df['rating'].mean()
print(f"Global average rating: {global_mean:.2f}")
print(f"Sparse matrix: {R_sparse.nnz:,} non-zero entries")

print("\n" + "=" * 60)
print("PART C: Matrix Factorization using SVD")
print("=" * 60)

# Use smaller k for speed (20 instead of 50)
k = 20
print(f"Using k = {k} latent factors")
print("Computing sparse SVD (this may take a moment)...")

# Center the ratings by subtracting user means (for better predictions)
user_means = np.zeros(n_users)
for i in range(n_users):
    row = R_sparse.getrow(i)
    if row.nnz > 0:
        user_means[i] = row.data.mean()

# Create centered sparse matrix
R_centered = R_sparse.copy()
for i in range(n_users):
    row_start = R_centered.indptr[i]
    row_end = R_centered.indptr[i+1]
    R_centered.data[row_start:row_end] -= user_means[i]

# Perform sparse SVD
U, sigma, Vt = svds(R_centered.astype(float), k=k)

# Sort by singular values
idx = np.argsort(sigma)[::-1]
sigma = sigma[idx]
U = U[:, idx]
Vt = Vt[idx, :]

print(f"SVD complete!")
print(f"U shape: {U.shape}, Sigma: {sigma.shape}, Vt shape: {Vt.shape}")

# Reconstruct predictions: R_centered_pred = U @ diag(sigma) @ Vt
# Add user means back to get final predictions
sigma_diag = np.diag(sigma)
R_pred = np.dot(np.dot(U, sigma_diag), Vt)

# Add user means back
for i in range(n_users):
    R_pred[i, :] += user_means[i]

# Clip to valid range
R_pred = np.clip(R_pred, 1, 5)

print(f"Predictions range: [{R_pred.min():.2f}, {R_pred.max():.2f}]")

print("\n" + "=" * 60)
print("PART D: Predicting Ratings on Test Set")
print("=" * 60)

test_users_mapped = test_df['user_id'].map(user2idx)
test_items_mapped = test_df['item_id'].map(item2idx)

valid_mask = test_users_mapped.notna() & test_items_mapped.notna()

print(f"Cold start cases skipped: {(~valid_mask).sum():,}")

user_indices_test = test_users_mapped[valid_mask].astype(int).values
item_indices_test = test_items_mapped[valid_mask].astype(int).values

predicted_ratings = R_pred[user_indices_test, item_indices_test]
actual_ratings = test_df.loc[valid_mask, 'rating'].values

print(f"Total predictions made: {len(predicted_ratings):,}")

print("\n" + "=" * 60)
print("PART E: Evaluation Metrics")
print("=" * 60)

mae = np.mean(np.abs(predicted_ratings - actual_ratings))
rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))

print(f"\n*** EVALUATION RESULTS ***")
print(f"MAE  (Mean Absolute Error):           {mae:.4f}")
print(f"RMSE (Root Mean Squared Error):       {rmse:.4f}")

print("\n" + "=" * 60)
print("PART F: Saving Results")
print("=" * 60)

results_df = pd.DataFrame({
    'user_id': test_df.loc[valid_mask, 'user_id'].values,
    'item_id': test_df.loc[valid_mask, 'item_id'].values,
    'actual_rating': actual_ratings,
    'predicted_rating': np.round(predicted_ratings, 2)
})

results_df.to_csv('data/rating_predictions.csv', index=False)
print(f"Saved: data/rating_predictions.csv ({len(results_df):,} rows)")

np.save('data/R_pred.npy', R_pred)
print(f"Saved: data/R_pred.npy (shape: {R_pred.shape})")

with open('data/user2idx.pkl', 'wb') as f:
    pickle.dump(user2idx, f)
print(f"Saved: data/user2idx.pkl")

with open('data/item2idx.pkl', 'wb') as f:
    pickle.dump(item2idx, f)
print(f"Saved: data/item2idx.pkl")

print("\n" + "=" * 60)
print("Step 2 Complete!")
print("=" * 60)
