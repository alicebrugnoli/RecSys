from main import *
from Evaluation.Evaluator import *
from Data_manager.split_train_validation_random_holdout import *
from Recommender.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

# Replacing 1 with 0 and viceversa because when 0 the user has interacted with the item
interactions_and_impressions["Data"] = interactions_and_impressions["Data"].replace([0, 1])

# SPARSE MATRIX
# See that the max ID of items and users is higher than the number of unique values -> empty profiles
# We should remove empty indices, to do so we create a new mapping

mapped_id, original_id = pd.factorize(interactions_and_impressions["UserID"].unique())
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)
mapped_id, original_id = pd.factorize(interactions_and_impressions["ItemID"].unique())
item_original_ID_to_index = pd.Series(mapped_id, index=original_id)


# We now replace the IDs in the dataframe and we are ready to use the data.
interactions_and_impressions["UserID"] = interactions_and_impressions["UserID"].map(user_original_ID_to_index)
interactions_and_impressions["ItemID"] = interactions_and_impressions["ItemID"].map(item_original_ID_to_index)


URM_all = sps.coo_matrix((interactions_and_impressions["Data"].values,
                         (interactions_and_impressions["UserID"].values, interactions_and_impressions["ItemID"].values)))

URM_all = URM_all.tocsr()

# SPLITTING
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

#evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
#evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

"""""
class ItemKNNCFRecommender(object):
    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink, topK=topK, normalize=normalize,
                                                      similarity=similarity)
        W_sparse = similarity_object.compute_similarity()
        return W_sparse

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.fit()).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]
        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


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


# We pass as paramether the recommender class
def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_AP = 0.0
    num_eval = 0
    for user_id in range(URM_test.shape[0]):
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]
        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1
            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += AP(recommended_items, relevant_items)
    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    MAP = cumulative_AP / num_eval
    print("Recommender results are: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, MAP))
"""

class UserKNNCFRecommender:

    def __init__(self, URM):
        self.URM = URM
        self.W_sparse = self.fit()

    def fit(self, topK=15, shrink=10, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM.T, shrink=shrink, topK=topK, normalize=normalize,
                                                      similarity=similarity)
        W_sparse = similarity_object.compute_similarity()
        return W_sparse

    def recommend(self, user_id, at, exclude_seen=True):
        # compute the scores using the dot product
        scores = self.W_sparse[user_id, :].dot(self.URM).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]
        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


def write_submission(user_id, items, f):
    f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")


recommender = UserKNNCFRecommender(URM_train)
with open("./submission.csv", "w") as f:
    for user_id in range(URM_test.shape[0]):
        write_submission(user_id, recommender.recommend(user_id, 10), f)

#evaluate_algorithm(URM_test, recommender)
