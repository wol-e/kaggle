from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys

sys.path.append("...")  # TODO: to find the config. there must be a better way.
from config import TRAINING_FEATURES


class FeaturePipeline():
    def __init__(self):
        self.encoder = Encoder()
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, df):
        df = df[TRAINING_FEATURES]
        df = nan_imputation(df)
        self.encoder.fit(df)

    def transform(self, df):
        df = df[TRAINING_FEATURES]
        df = nan_imputation(df)
        df = self.encoder.transform(df)
        df = self.scaler.fit_transform(df)

        return df


class Encoder:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, df):
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)

        self.encoder.fit(df)

    def transform(self, df):
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)
        return self.encoder.transform(df)


def nan_imputation(df):
    return df.fillna("NONE")
