import numpy as np
from sklearn.preprocessing import OrdinalEncoder

import sys

sys.path.append("...")  # TODO: to find the config and other sources, is there a nicer way?

from helpers.memory import reduce_memory_usage
TRAINING_FEATURES = ["Asset_ID", "Count", "Open", "High", "Low", "Close", "Volume", "VWAP"]
OUTPUT_FEATURES = ["Asset_ID", "Count", "Open", "High", "Low", "Close", "Volume", "VWAP", "Upper_Shadow", "Lower_Shadow"]

class FeaturePipeline():
    def __init__(self):
        pass

    def fit(self, df):
        pass

    def transform(self, df):
        df = reduce_memory_usage(df)
        df = df[TRAINING_FEATURES]
        # df = nan_imputation(df)

        def upper_shadow(df):
            return df['High'] - np.maximum(df['Close'], df['Open'])

        def lower_shadow(df):
            return np.minimum(df['Close'], df['Open']) - df['Low']

        df['Upper_Shadow'] = upper_shadow(df)
        df['Lower_Shadow'] = lower_shadow(df)


        return df[OUTPUT_FEATURES]

def nan_imputation(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0)
