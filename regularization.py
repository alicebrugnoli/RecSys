import os

from skopt.space import Integer, Categorical, Real

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommender.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommender.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from varie import initDataSet

if __name__ == '__main__':
    URM_all, users_to_recommend, ICM_length, ICM_type = initDataSet()
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    evaluator_validation = EvaluatorHoldout(URM_validation, [10], exclude_seen=True)
    recommender = RP3betaRecommender
    hyperparameters_range_dictionary = {
        "topK": Integer(20, 120),
        "alpha": Real(0.3, 0.7),
        "beta": Real(0.0, 0.4),
        "normalize_similarity": Categorical([True])
    }

    hyperparameterSearch = SearchBayesianSkopt(recommender,
                                               evaluator_validation=evaluator_validation)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],  # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_all],
        # For a CBF model simply put [URM_train_validation, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

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
                                output_file_name_root=recommender.RECOMMENDER_NAME,  # How to call the files
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                )

