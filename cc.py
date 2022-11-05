import pandas as pd
import scipy.sparse as sps
import numpy as np
import time

from Recommender.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


def conv(x):
    return [word.strip() for word in x.split(',')]


def initDataSet():
    # add path files to csv
    user_interaction_file_path = '/Users/alicebrugnoli/Desktop/dataset Recsys/interactions_and_impressions.csv'

    converter = conv
    # This is the matrix of the interactions
    URM_all_dataframe = pd.read_csv(user_interaction_file_path,
                                    sep=",",
                                    dtype={0: int, 1: int, 3: int},
                                    converters={2: converter})

    # Mapping of the matrix column
    URM_all_dataframe.columns = ["UserID", "ItemID", "Impressions", "Interactions"]
    # Replacing 1 with 0 and viceversa because when 0 the user has interacted with the item
    URM_all_dataframe["Interactions"] = URM_all_dataframe["Interactions"].replace([0, 1])
    URM_all_dataframe.drop('Impressions', axis=1, inplace=True)
    # URM_all_dataframe.sort_values(by=['Interactions'], ascending=False, inplace=True)
    URM_all_dataframe.drop_duplicates(subset=['UserID', 'ItemID'], keep='first', inplace=True)

    unique_users = URM_all_dataframe.UserID.unique()
    unique_items = URM_all_dataframe.ItemID.unique()

    num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
    num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

    mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "UserID": unique_users})
    mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "ItemID": unique_items})

    URM_all_dataframe = pd.merge(left=URM_all_dataframe,
                       right=mapping_user_id,
                       how="inner",
                       on="UserID")

    URM_all_dataframe = pd.merge(left=URM_all_dataframe,
                       right=mapping_item_id,
                       how="inner",
                       on="ItemID")

    # The COO constructor expects (data, (row, column))
    URM_all = sps.coo_matrix((URM_all_dataframe["Interactions"].values,
                              (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)))

    # URM_all = URM_all.tocsr()
    return URM_all, unique_users, mapping_user_id, mapping_item_id, URM_all_dataframe


def splittingData(URM):
    train_test_split = 0.60

    n_interactions = URM.nnz

    train_mask = np.random.choice([True, False], n_interactions, p=[train_test_split, 1 - train_test_split])
    URM_train = sps.csr_matrix((URM.data[train_mask], (URM.row[train_mask], URM.col[train_mask])))

    test_mask = np.logical_not(train_mask)
    URM_test = sps.csr_matrix((URM.data[test_mask], (URM.row[test_mask], URM.col[test_mask])))
    return URM_train, URM_test


def prepare_submission(ratings, users_to_recommend, urm_train, recommender):
    users_ids_and_mappings = ratings[["UserID", "mapped_user_id"]].drop_duplicates()
    items_ids_and_mappings = ratings[["ItemID", "mapped_item_id"]].drop_duplicates()

    mapping_to_item_id = dict(zip(ratings.mapped_item_id, ratings.ItemID))

    recommendation_length = 10
    submission = []
    start_time = time.time()
    for idx, row in users_ids_and_mappings.iterrows():
        user_id = row.UserID
        mapped_user_id = row.mapped_user_id

        recommendations = recommender.recommend(user_id=mapped_user_id,
                                                at=recommendation_length,
                                                exclude_seen=True)

        submission.append((user_id, [mapping_to_item_id[item_id] for item_id in recommendations]))
    end_time = time.time()
    print("Reasonable implementation speed is {:.2f} usr/sec".format(users_to_recommend.size / (end_time - start_time)))

    return submission


def write_submission(submissions):
    with open("./submission.csv", "w") as f:
        for user_id, items in submissions:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")


def precision(recommended_items, relevant_items):
    # array([ 241, 1622,   15,  857, 5823]) recommended items
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # array([False, True, False, True, False]) if there is a correspondance of an element of recommended items in relevant items
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


def recall(recommended_items, relevant_items):
    # array([ 241, 1622,   15,  857, 5823]) recommended items
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # array([False, True, False, True, False]) if there is a correspondance of an element of recommended items in relevant items
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0] # recall is based on the number of relevant items
    return recall_score


def AP(recommended_items, relevant_items):
    # recommended_items = np.array([241, 1622, 15, 857, 5823])
    # relevant_items = np.array([   5,   18,   20,   29,   30,   79,   83,  139,  166,  179,  241,
    #         413, 1280, 1303, 1310, 1312, 1429, 1622, 2310, 2492])
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    # print(is_relevant * np.cumsum(is_relevant, dtype=np.float32))  [1. 2. 0. 0. 0.]
    # print(1 + np.arange(is_relevant.shape[0]))  [1 2 3 4 5]
    # print(p_at_k)  [1. 1. 0. 0. 0.]
    ap_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    # print(ap_score)  0.4
    return ap_score


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

            cumulative_precision += precision(recommended_items, relevant_items) # this is done for a single user
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += AP(recommended_items, relevant_items)

    cumulative_precision /= num_eval  # this is the average of the entire user set
    cumulative_recall /= num_eval
    MAP = cumulative_AP / num_eval

    print("Recommender results are: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, MAP))

class ItemKNNCFRecommender(object):
    def __init__(self, URM):
        self.URM = URM
        self.W_sparse = self.fit()

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink, topK=topK, normalize=normalize,
                                                      similarity=similarity)
        W_sparse = similarity_object.compute_similarity()
        return W_sparse

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
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


if __name__ == '__main__':
    URM_all, users, mapping_user_id, mapping_item_id, DataSet = initDataSet()
    URM_train, URM_test = splittingData(URM_all)
    recommender = ItemKNNCFRecommender(URM_train)
    #recommender.fit(shrink=10, topK=15) #Hyper parameters
    evaluate_algorithm(URM_test, recommender, at=10)
    submission = prepare_submission(DataSet, users, URM_all, recommender)
    write_submission(submission)




