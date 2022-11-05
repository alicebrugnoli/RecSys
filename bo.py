from main import *
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.sparse as sps

# See that the max ID of items and users is higher than the number of unique values -> empty profiles
# We should remove empty indices, to do so we create a new mapping
mapped_id, original_id = pd.factorize(interactions_and_impressions["UserID"].unique())
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)
mapped_id, original_id = pd.factorize(interactions_and_impressions["ItemID"].unique())
item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

# We now replace the IDs in the dataframe and we are ready to use the data.
interactions_and_impressions["UserID"] = interactions_and_impressions["UserID"].map(user_original_ID_to_index)
interactions_and_impressions["ItemID"] = interactions_and_impressions["ItemID"].map(item_original_ID_to_index)

userID_unique = interactions_and_impressions["UserID"].unique()
itemID_unique = interactions_and_impressions["ItemID"].unique()

URM_all = sps.coo_matrix((interactions_and_impressions["Data"].values,
                         (interactions_and_impressions["UserID"].values, interactions_and_impressions["ItemID"].values)))

URM_all.tocsr()

# We compute the item popularity as the number of interaction in each column
# We can use the properties of sparse matrices in CSC format
item_popularity = np.ediff1d(URM_all.tocsc().indptr)
item_popularity = np.sort(item_popularity)

# SPLITTING THE DATA
train_test_split = 0.80
n_interactions = URM_all.nnz
train_mask = np.random.choice([True, False], n_interactions, p=[train_test_split, 1-train_test_split])

URM_train = sps.csr_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
test_mask = np.logical_not(train_mask)

URM_test = sps.csr_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))


def precision(recommended_items, relevant_items):  # Precision: how many of the recommended items are relevant
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


def recall(recommended_items, relevant_items):  # Recall: how many of the relevant items I was able to recommend
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


def AP(recommended_items, relevant_items):  # Average Precision
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    ap_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return ap_score
