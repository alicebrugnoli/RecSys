import pandas as pd
import numpy as np


# remove ,
def converter(x):
    return [word.strip() for word in x.split(',')]


converters = {"Impressions": converter}
interactions_and_impressions = pd.read_csv(
    '/Users/alicebrugnoli/Desktop/dataset Recsys/interactions_and_impressions.csv',
    converters=converters)
interactions_and_impressions.columns = ["UserID", "ItemID", "Impressions", "Data"]

data_ICM_length = pd.read_csv('/Users/alicebrugnoli/Desktop/dataset Recsys/data_ICM_length.csv',
                              dtype={0: int, 1: int, 2: int},
                              engine='python')
data_ICM_length.columns = ["ItemID", "FeatureID", "Data"]

data_ICM_type = pd.read_csv('/Users/alicebrugnoli/Desktop/dataset Recsys/data_ICM_type.csv',
                            dtype={0: int, 1: int, 2: int},
                            engine='python')
data_ICM_length.columns = ["ItemID", "FeatureID", "Data"]
