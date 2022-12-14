from skopt.space import Integer, Categorical, Real
from Evaluation.Evaluator import EvaluatorHoldout
from varie import *
from Recommender.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
import os

if __name__ == '__main__':
    URM_all, data_target_users_test, ICM_length, ICM_type = initDataSet()
    URM_train, URM_validation, URM_test, URM_train_validation = splittingData_in_three(URM_all)
    evaluator_validation = EvaluatorHoldout(URM_validation, [10], exclude_seen=True)
    evaluator_test = EvaluatorHoldout(URM_test, [10], exclude_seen=True)
    recommender_class = SLIM_BPR_Cython
    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)
    hyperparameters_range_dictionary = {
        "epochs": Categorical([30]),
        "topK": Categorical([50, 100, 200, 500]),
        "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
        "learning_rate": Categorical([1e-3, 1e-4, 1e-5, 1e-6])
    }

    earlystopping_keywargs = {"validation_every_n": 30,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation,
                              "lower_validations_allowed": 30,
                              "validation_metric": "MAP",
                              }

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],  # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=earlystopping_keywargs
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation],
        # For a CBF model simply put [URM_train_validation, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=earlystopping_keywargs
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
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize
                                )
