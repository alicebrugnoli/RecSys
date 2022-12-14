import csv

from numpy import linalg as LA
import numpy as np
import time
from tqdm import tqdm

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommender.BaseRecommender import BaseRecommender
from Recommender.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommender.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommender.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommender.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommender.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommender.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommender.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommender.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender, \
    ScaledPureSVDRecommender
from Recommender.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from varie import initDataSet
import scipy.sparse as sps


class ScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute):
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights


class DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, norm, alpha=0.5):

        self.alpha = alpha
        self.norm = norm

    def _compute_item_score(self, user_id_array, items_to_compute):

        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)

        if norm_item_weights_1 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

        if norm_item_weights_2 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (
                    1 - self.alpha)

        return item_weights



def stacked(URM_train, ICM_genres):
    stacked_URM = sps.vstack([URM_train, ICM_genres.T])
    stacked_URM = sps.csr_matrix(stacked_URM)
    stacked_ICM = sps.csr_matrix(stacked_URM.T)

    return stacked_URM, stacked_ICM

def hybrid_models_with_the_same_structure(URM):
    P3alpha = P3alphaRecommender(URM)
    P3alpha.fit(topK=100, alpha=1.0, normalize_similarity=True)
    RP3beta = RP3betaRecommender(URM)
    RP3beta.fit(topK=89, alpha=0.7, beta=0.4, normalize_similarity=True)
    alpha = 0.5
    new_similarity = (1 - alpha) * P3alpha.W_sparse + alpha * RP3beta.W_sparse
    return new_similarity

def hybrid_of_models_with_different_structure(URM):
    pureSVD = PureSVDRecommender(URM)
    pureSVD.fit()
    itemKNNCF = ItemKNNCFRecommender(URM)
    itemKNNCF.fit()
    return itemKNNCF, pureSVD

def models_with_rating_prediction_vs_ranking_loss_functions(URM):
    slim_bpr_recommender = SLIM_BPR_Cython(URM)
    slim_bpr_recommender.fit(epochs=200, topK=100, sgd_mode='adagrad', learning_rate=0.0001)
    recommender_object = RP3betaRecommender(URM)
    recommender_object.fit(topK=100, alpha=0.5, beta=0.2, normalize_similarity=True)
    return slim_bpr_recommender, recommender_object


def conv(x):
    return [word.strip() for word in x.split(',')]


def splittingData(URM):
    train_test_split = 0.1

    n_interactions = URM.nnz

    train_mask = np.random.choice([True, False], n_interactions, p=[train_test_split, 1 - train_test_split])
    URM_train = sps.csr_matrix((URM.data[train_mask], (URM.row[train_mask], URM.col[train_mask])))

    test_mask = np.logical_not(train_mask)
    URM_test = sps.csr_matrix((URM.data[test_mask], (URM.row[test_mask], URM.col[test_mask])))
    return URM_train, URM_test


def prepare_submission_no_mapping(users_to_recommend, recommender):
    recommendation_length = 10
    submission = []
    start_time = time.time()
    for row in tqdm(np.nditer(users_to_recommend)):
        user_id = int(row)
        recommendations = recommender.recommend(user_id_array=user_id,
                                                at=recommendation_length)
        submission.append((user_id, recommendations))
    end_time = time.time()
    print("Reasonable implementation speed is {:.2f} usr/sec".format(users_to_recommend.size / (end_time - start_time)))

    return submission


def write_submission(f, user_id, items):
    f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")


if __name__ == '__main__':
    URM_all, users_to_recommend, ICM_length, ICM_type = initDataSet()
    #URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    recommender = RP3betaRecommender(URM_all)
    recommender.fit(topK=89, alpha=0.7, beta=0.4, normalize_similarity=True)
    submission = prepare_submission_no_mapping(users_to_recommend, recommender)
    with open("./submission.csv", "w") as f:
        file = csv.DictWriter(f, fieldnames=["user_id", "item_list"])
        file.writeheader()
        blank_users = 0
        for user_id, items in tqdm(submission):
            write_submission(f, user_id, items)
    """
    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    result_df, _ = evaluator.evaluateRecommender(recommender_object)
    print("Result: {}".format(result_df.loc[10]["MAP"]))
    for norm in [1, 2, np.inf, -np.inf]:
        recommender.fit(norm, alpha=0.8)
        result_df, _ = evaluator.evaluateRecommender(recommender)
        print("Norm: {}, Result: {}".format(norm, result_df.loc[10]["MAP"]))
    submission = prepare_submission_no_mapping(users_to_recommend, recommender)
    with open("./submission.csv", "w") as f:
        file = csv.DictWriter(f, fieldnames=["user_id", "item_list"])
        file.writeheader()
        blank_users = 0
        for user_id, items in tqdm(submission):
            write_submission(f, user_id, items)
    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    result_df, _ = evaluator.evaluateRecommender(recommender)
    print("Norm: {}, Result: {}".format(1, result_df.loc[10]["MAP"]))
    submission = prepare_submission_no_mapping(users_to_recommend, recommender)
    with open("./submission.csv", "w") as f:
        file = csv.DictWriter(f, fieldnames=["user_id", "item_list"])
        file.writeheader()
        blank_users = 0
        for user_id, items in tqdm(submission):
            write_submission(f, user_id, items)
    submission = prepare_submission_no_mapping(users_to_recommend, recommender_object)
    with open("./submission.csv", "w") as f:
        file = csv.DictWriter(f, fieldnames=["user_id", "item_list"])
        file.writeheader()
        blank_users = 0
        for user_id, items in tqdm(submission):
            write_submission(f, user_id, items)
    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)
    print("Algorithm: {}, results: \n{}".format(recommender_object, results_run_string_1))
    slim_bpr_recommender, nmf_recommender = models_with_rating_prediction_vs_ranking_loss_functions(URM_train)
    recommender = DifferentLossScoresHybridRecommender(URM_train, slim_bpr_recommender, nmf_recommender)
    for norm in [1, 2, np.inf, -np.inf]:
        recommender.fit(norm, alpha=0.8)
        result_df, _ = evaluator.evaluateRecommender(recommender)
        print("Norm: {}, Result: {}".format(norm, result_df.loc[10]["MAP"]))
    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender)
    print("Algorithm: {}, results: \n{}".format(recommender, results_run_string_1))
    submission = prepare_submission_no_mapping(users_to_recommend, recommender)
    with open("./submission.csv", "w") as f:
        file = csv.DictWriter(f, fieldnames=["user_id", "item_list"])
        file.writeheader()
        blank_users = 0
        for user_id, items in tqdm(submission):
            write_submission(f, user_id, items)
            
    new_similarity = hybrid_models_with_the_same_structure(URM_train)
    itemKNNCF, pureSVD = hybrid_of_models_with_different_structure(URM_train)
    recommender = ScoresHybridRecommender(URM_train, itemKNNCF, pureSVD)
    recommender.fit()
    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender)
    print("Algorithm: {}, results: \n{}".format(recommender, results_run_string_1))
    """
