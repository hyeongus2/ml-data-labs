import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# --- Hyperparameters ---
# Tune these values using a validation set by checking the best ROC-AUC score

# Number of latent factors to use in SVD
K = 11

RANDOM_SEED = 42
# -----------------------


def load_and_build_utility_matrix(df_train):
    """
    Load training data and build the implicit utility matrix R.
    """
    # Get all unique users and movies from the training set
    all_user_ids = df_train['user_id'].unique()
    all_movie_ids = df_train['movie_id'].unique()

    user_map = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
    movie_map = {movie_id: idx for idx, movie_id in enumerate(all_movie_ids)}

    # Build the utility matrix R
    R = np.zeros((len(all_user_ids), len(all_movie_ids)))

    # Fill the utility matrix with implicit feedback (1 for rated movies)
    rows = df_train['user_id'].map(user_map).values
    cols = df_train['movie_id'].map(movie_map).values

    R[rows, cols] = 1
    return R, user_map, movie_map


def compute_bias_terms(R):
    """
    Compute user and movie bias terms from the utility matrix R.
    mu: global mean
    bu: user bias
    bi: movie bias
    """
    U, M = R.shape  # Users, Movies
    # Global mean
    mu = np.sum(R) / (U * M)
    # User bias (shape: U,)
    bu = (np.sum(R, axis=1) / M) - mu
    # Movie bias (shape: M,)
    bi = (np.sum(R, axis=0) / U) - mu
    return mu, bu, bi


def train_svd_model(R, n_components):
    """
    Train SVD model on the utility matrix R.
    """
    svd_model = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)

    # U_factors = U * S
    U_factors = svd_model.fit_transform(R)
    # V_T_factors = V^T
    V_T_factors = svd_model.components_
    return U_factors, V_T_factors


def make_predictor(mu, bu, bi, U_factors, V_T_factors, user_map, movie_map):
    """
    Create a prediction function that uses bias terms and SVD factors.
    """

    def predict_score(user_id, movie_id):
        known_user = user_id in user_map
        known_movie = movie_id in movie_map

        # Start with global mean
        score = mu

        # Add user bias if user is known
        if known_user:
            user_idx = user_map[user_id]
            score += bu[user_idx]

        # Add movie bias if movie is known
        if known_movie:
            movie_idx = movie_map[movie_id]
            score += bi[movie_idx]

        # Add interaction term if both user and movie are known
        if known_user and known_movie:
            user_idx = user_map[user_id]
            movie_idx = movie_map[movie_id]
            score += np.dot(U_factors[user_idx, :], V_T_factors[:, movie_idx])

        return score

    return predict_score


if __name__ == "__main__":
    # Train Data set is argument 1
    train_data = sys.argv[1]
    # Test Data set is argument 2
    test_data = sys.argv[2]

    # Load training and test data
    df_train = pd.read_csv(train_data,
                           sep=',',
                           header=None,
                           names=['user_id', 'movie_id', 'rating', 'timestamp'])

    df_test = pd.read_csv(test_data,
                          sep=',',
                          header=None,
                          names=['user_id', 'movie_id', 'timestamp'])

    # Load data and build utility matrix
    R, user_map, movie_map = load_and_build_utility_matrix(df_train)

    # Compute bias terms
    mu, bu, bi = compute_bias_terms(R)

    # Train SVD and get predicted utility matrix
    U_factors, V_factors_T = train_svd_model(R, K)

    predict_score = make_predictor(mu, bu, bi, U_factors, V_factors_T, user_map, movie_map)
    
    # Output file name is output3c.txt
    with open('output3c.txt', 'w') as out:
        for row in df_test.itertuples():
            # <USER ID>,<MOVIE ID>,<TIMESTAMP>
            user_id = row.user_id
            movie_id = row.movie_id
            timestamp = row.timestamp

            # <USER ID>,<MOVIE ID>,<SCORE FOR MOVIE><TIMESTAMP>
            score = predict_score(user_id, movie_id)
            out.write(f"{user_id},{movie_id},{score},{timestamp}\n")