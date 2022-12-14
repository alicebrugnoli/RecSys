import csv
from tqdm import tqdm

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommender.MatrixFactorization.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython, MatrixFactorization_BPR_Cython
from varie import initDataSet, prepare_submission, write_submission, splittingData_in_two, splittingData_in_three

if __name__ == '__main__':
    URM_all, users_to_recommend = initDataSet()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    recommender = MatrixFactorization_FunkSVD_Cython(URM_train)
    recommender.fit()
    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender)
    print("Algorithm: {}, results: \n{}".format(recommender, results_run_string_1))
    submission = prepare_submission(users_to_recommend, recommender)
    with open("./submission.csv", "w") as f:
        file = csv.DictWriter(f, fieldnames=["user_id", "item_list"])
        file.writeheader()
        blank_users = 0
        for user_id, items in tqdm(submission):
            write_submission(user_id, items, f)
    """"
    hyperparameters_range_dictionary = {
        "epochs": Categorical([1]),
        "num_factors": Integer(1, 200),
        "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
        "batch_size": Categorical([8, 16, 32, 64, 128, 256, 512, 1024]),
        "item_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
        "user_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
        "learning_rate": Categorical([1e-3, 1e-4, 1e-5, 1e-6]),
    }

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],  # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation],
        # For a CBF model simply put [URM_train_validation, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_class = MatrixFactorization_FunkSVD_Cython

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)
    output_folder_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 10  # using 10 as an example
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                save_model="last",
                                output_folder_path=output_folder_path,  # Where to save the results
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize)
    """
