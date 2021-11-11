import numpy as np
import pandas as pd

import sys

sys.path.append("...")  # TODO: to find the config and other sources, is there a nicer way?

from helpers.memory import reduce_memory_usage

TRAINING_FEATURES = ["Asset_ID", "Count", "Open", "High", "Low", "Close", "Volume", "VWAP"]
OUTPUT_FEATURES = ["Asset_ID", "Count", "Open", "High", "Low", "Close",
                   "Volume", "VWAP", "Upper_Shadow", "Lower_Shadow", "open2close", "high2low",
                   "high2mean",
                   "low2mean",
                   "high2median",
                   "low2median",
                   "volume2count",
                   ]


class FeaturePipeline():
    def __init__(self):
        pass

    def fit(self, df):
        pass

    def transform(self, df):
        df = df[TRAINING_FEATURES]
        # df = nan_imputation(df)

        pd.options.mode.chained_assignment = None  # default='warn'
        # inspired by https://www.kaggle.com/code1110/gresearch-simple-lgb-starter
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
        df['Lower_Shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
        df['open2close'] = df['Close'] / df['Open']
        df['high2low'] = df['High'] / df['Low']
        mean_price = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        median_price = df[['Open', 'High', 'Low', 'Close']].median(axis=1)
        df['high2mean'] = df['High'] / mean_price
        df['low2mean'] = df['Low'] / mean_price
        df['high2median'] = df['High'] / median_price
        df['low2median'] = df['Low'] / median_price
        df['volume2count'] = df['Volume'] / (df['Count'] + 1)

        df = reduce_memory_usage(df)

        return df[OUTPUT_FEATURES]


def nan_imputation(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0)
