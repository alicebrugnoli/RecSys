import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample


# remove ,
def conv(x):
    return [word.strip() for word in x.split(',')]


def initDataSet():
    # add path files to csv
    user_interaction_file_path = '/Users/alicebrugnoli/Desktop/dataset Recsys/interactions_and_impressions.csv'
    user_to_recommend_file_path = '/Users/alicebrugnoli/Desktop/dataset Recsys/data_target_users_test.csv'
    data_ICM_length_file_path = '/Users/alicebrugnoli/Desktop/dataset Recsys/data_ICM_length.csv'
    data_ICM_type_file_path = '/Users/alicebrugnoli/Desktop/dataset Recsys/data_ICM_type.csv'

    converter = conv
    # This is the matrix of the interactions
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=user_interaction_file_path,
                                    sep=",",
                                    dtype={0: int, 1: int, 3: int},
                                    converters={2: converter})
    data_target_users_test = pd.read_csv(filepath_or_buffer=user_to_recommend_file_path,
                                         dtype={0: int})
    ICM_length = pd.read_csv(filepath_or_buffer=data_ICM_length_file_path,
                             dtype={0: int, 1: int, 2: int})
    ICM_length.columns = ["ItemID", "FeatureID", "Data"]

    ICM_type = pd.read_csv(filepath_or_buffer=data_ICM_type_file_path,
                           dtype={0: int, 1: int, 2: int})

    ICM_type.columns = ["ItemID", "FeatureID", "Data"]

    # Mapping of the matrix column
    URM_all_dataframe.columns = ["UserID", "ItemID", "Impressions", "Interactions"]
    data_target_users_test.columns = ["UserID"]

    # Replacing 1 with 0 and viceversa because when 0 the user has interacted with the item
    URM_all_dataframe["Interactions"] = URM_all_dataframe["Interactions"].replace([0, 1])
    URM_all_dataframe.drop('Impressions', axis=1, inplace=True)
    URM_all_dataframe.sort_values(by=['Interactions'], ascending=False, inplace=True)
    URM_all_dataframe.drop_duplicates(subset=['UserID', 'ItemID'], keep='first', inplace=True)

    ICM_length = ICM_length.loc[ICM_length["ItemID"].isin(URM_all_dataframe["ItemID"])]
    ICM_type = ICM_type.loc[ICM_type["ItemID"].isin(URM_all_dataframe["ItemID"])]

    num_users = max(URM_all_dataframe["UserID"].values) + 1
    num_items = max(URM_all_dataframe["ItemID"].values) + 1

    num_items_length = max(ICM_length["ItemID"].values) + 1
    num_features_length = max(ICM_length["FeatureID"].values) + 1

    num_items_type = max(ICM_type["ItemID"].values) + 1
    num_features_type = max(ICM_type["FeatureID"].values) + 1

    print(num_items, num_items_length, num_items_type)

    # The COO constructor expects (data, (row, column))
    URM_all = sps.coo_matrix((URM_all_dataframe["Interactions"].values,
                              (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)),
                             shape=(num_users, num_items))

    ICM_length = sps.coo_matrix((ICM_length["Data"].values,
                                 (ICM_length["ItemID"].values, ICM_length["FeatureID"].values)),
                                shape=(num_items_length, num_features_length))

    ICM_type = sps.coo_matrix((ICM_type["Data"].values,
                               (ICM_type["ItemID"].values, ICM_type["FeatureID"].values)),
                              shape=(num_items_type, num_features_type))
    URM_all = URM_all.tocsr()
    return URM_all, data_target_users_test, ICM_length, ICM_type


def splittingData_in_two(URM):
    train_test_split = 0.60
    n_interactions = URM.nnz
    train_mask = np.random.choice([True, False], n_interactions, p=[train_test_split, 1 - train_test_split])
    URM_train = sps.csr_matrix((URM.data[train_mask], (URM.row[train_mask], URM.col[train_mask])))
    test_mask = np.logical_not(train_mask)
    URM_test = sps.csr_matrix((URM.data[test_mask], (URM.row[test_mask], URM.col[test_mask])))
    return URM_train, URM_test


def splittingData_in_three(URM_all):
    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)
    return URM_train, URM_validation, URM_test, URM_train_validation


def prepare_submission(users_to_recommend, recommender):
    submission = []
    for row in tqdm(np.nditer(users_to_recommend)):
        user_id = int(row)
        recommendations = recommender.recommend(user_id, at=10)
        submission.append((user_id, recommendations))
    return submission


def write_submission(user_id, items, f):
    f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")