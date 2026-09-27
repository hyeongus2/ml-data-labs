import sys
import numpy as np

# Take average of top-k similar user's ratings
topk_users_to_average = 10
# Take average of top-k similar items ratings
topk_items_to_average = 10
# Considering items 1 to 1000
num_items_for_prediction = 1000
# Top-k predictions of items with highest ratings
topk_items = 5
# Target user's id
target_user_id = 600


def cosine(a, b):
    """
    INPUT: two vectors a and b
    OUTPUT: cosine similarity between a and b

    DESCRIPTION:
    Takes two vectors and returns the cosine similarity.
    """
    dot_product = np.dot(a, b)
    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def get_matrix(file_name):
    """
    INPUT: file name
    OUTPUT: utility matrix from the file

    DESCRIPTION:
    Reads the utility matrix from the file. 
    """
    # Fill the array with np.nan for missing values
    max_user_id = 0
    max_item_id = 0
    data = []

    with open(file_name, 'r') as f:
        for line in f:
            user_id, item_id, rating, _ = line.split(',')
            user_id = int(user_id)
            item_id = int(item_id)
            rating = float(rating)

            if user_id > max_user_id:
                max_user_id = user_id
            if item_id > max_item_id:
                max_item_id = item_id

            data.append((user_id, item_id, rating))

    # Initialize utility matrix with np.nan
    # +1 because user_id and item_id are 1-based
    umatrix = np.full((max_user_id + 1, max_item_id + 1), np.nan)
    for user_id, item_id, rating in data:
        umatrix[user_id, item_id] = rating
    return umatrix


def user_based(umatrix, user_id):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using user-based collaborative
    filtering.
    """
    # Compute mean ratings for each user, ignoring NaNs
    # keepdims=True makes it (num_users, 1) shape for broadcasting
    user_means = np.nanmean(umatrix, axis=1, keepdims=True)

    # Normalize the utility matrix by subtracting user means
    umatrix_norm = umatrix - user_means

    # Fill NaNs with 0 for similarity computation
    umatrix_norm_filled = np.nan_to_num(umatrix_norm, nan=0.0)

    # Compute cosine similarities between target user and all other users
    target_user_vector = umatrix_norm_filled[user_id]
    similarities = np.array([cosine(target_user_vector, umatrix_norm_filled[i]) for i in range(umatrix_norm_filled.shape[0])])
    similarities[user_id] = -1.0  # Exclude the target user from their own recommendations

    # Get the indices of the top-k most similar users
    top_k_users = np.argsort(similarities)[::-1][:topk_users_to_average]

    # Compute average ratings for each item based on top-k similar users
    # Consider only items rated by top-k similar users
    recommended_items = {}
    for item_id in range(1, num_items_for_prediction + 1):
        if item_id >= umatrix.shape[1]:
            break

        total_rating = 0.0
        count = 0
        for similar_user in top_k_users:
            rating = umatrix[similar_user, item_id]
            if not np.isnan(rating):
                total_rating += rating
                count += 1
        if count > 0:
            recommended_items[item_id] = total_rating / count

    # Get top-k items with highest predicted ratings
    # Sort by score in descending order, then by item_id in ascending order for tie-breaking
    top_k_recommendations = sorted(recommended_items.items(), key=lambda x: (-x[1], x[0]))[:topk_items]
    return top_k_recommendations


def item_based(umatrix, user_id):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using item-based collaborative
    filtering.
    """
    user_means = np.nanmean(umatrix, axis=1, keepdims=True)
    umatrix_norm = umatrix - user_means
    umatrix_norm_filled = np.nan_to_num(umatrix_norm, nan=0.0)

    # Transpose the matrix for item-based similarity
    umatrix_norm_filled_T = umatrix_norm_filled.T

    # Get the target user's original ratings
    target_user_vector = umatrix[user_id]
    # Indices of items rated by the target user
    target_user_rated_items_id = np.where(~np.isnan(target_user_vector))[0]

    predicted_ratings = {}

    # Define the range of other items to compare with (exclude items 1 to 1000)
    other_range_start = num_items_for_prediction + 1
    if other_range_start > umatrix_norm_filled_T.shape[0]:
        return []
    other_items_vectors = umatrix_norm_filled_T[other_range_start:]
    other_items_norms = np.sqrt(np.sum(other_items_vectors ** 2, axis=1))
    other_items_norms[other_items_norms == 0] = 1e-10  # Avoid division by zero

    for item_id in range(1, num_items_for_prediction + 1):
        if item_id >= umatrix_norm_filled_T.shape[0]:
            break

        # Compute similarities (matrix-vector multiplication)
        target_item_vector = umatrix_norm_filled_T[item_id]
        target_item_norm = np.sqrt(np.sum(target_item_vector ** 2))

        if target_item_norm == 0:
            continue

        dot_products = np.dot(other_items_vectors, target_item_vector)
        similarities = dot_products / (target_item_norm * other_items_norms)

        # Get top-k similar items
        top_k_items_indices_relative = np.argsort(similarities)[::-1][:topk_items_to_average]
        top_k_items_ids = top_k_items_indices_relative + other_range_start

        # Filter out items already rated by the target user
        other_items_mask = np.isin(top_k_items_ids, target_user_rated_items_id)
        filtered_top_k_items_ids = top_k_items_ids[other_items_mask]

        # Compute predicted rating for the item
        filtered_ratings = target_user_vector[filtered_top_k_items_ids]
        predicted_rating = np.mean(filtered_ratings) if filtered_ratings.size > 0 else 0.0
        predicted_ratings[item_id] = predicted_rating

    # Get top-k items with highest predicted ratings
    # Sort by score in descending order, then by item_id in ascending order for tie-breaking
    top_k_recommendations = sorted(predicted_ratings.items(), key=lambda x: (-x[1], x[0]))[:topk_items]
    return top_k_recommendations


if __name__ == "__main__":
    target_user_id = 600
    umatrix = get_matrix(sys.argv[1])
    ub_results = user_based(umatrix, target_user_id)

    with open('output3b_user.txt', 'w') as out:
        for item_id, score in ub_results:
            out.write(f"{item_id}\t{score}\n")
    
    ib_results = item_based(umatrix, target_user_id)
    with open('output3b_item.txt', 'w') as out:
        for item_id, score in ib_results:
            out.write(f"{item_id}\t{score}\n")