import csv

from Evaluation.Evaluator import EvaluatorHoldout
from varie import *
from Recommender.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

if __name__ == '__main__':
    URM_all, users_to_recommend = initDataSet()
    URM_train, URM_test = splittingData_in_two(URM_all)
    recommender = SLIM_BPR_Cython(URM_train)
    recommender.fit(epochs=300, topK=100, sgd_mode='adagrad', learning_rate=0.0001, )
    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender)
    print("Algorithm: {}, results: \n{}".format(recommender, results_run_string_1))
    """"
    submission = prepare_submission(users_to_recommend, recommender)
    with open("./submission.csv", "w") as f:
        file = csv.DictWriter(f, fieldnames=["user_id", "item_list"])
        file.writeheader()
        blank_users = 0
        for user_id, items in tqdm(submission):
            write_submission(user_id, items, f)
    """
