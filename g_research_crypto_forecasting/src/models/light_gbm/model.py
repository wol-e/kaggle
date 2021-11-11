import pandas as pd
import numpy as np
from .feature_engineering import OUTPUT_FEATURES
from lightgbm import LGBMRegressor
import sys
import time

sys.path.append("...")  # TODO: to find the config. there must be a better way.

from config import ASSET_DETAILS_PATH


class Model:
    def __init__(self):
        self._models = {}
        self.asset_details = pd.read_csv(ASSET_DETAILS_PATH).sort_values("Asset_ID")

    def fit(self, df, y):
        def get_model_for_asset(df, y, asset_id):
            df["y"] = y
            df = df[df["Asset_ID"] == asset_id]

            df = df.dropna(how="any")

            params = {
                'n_estimators': 100,
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'max_depth': -1,
                'learning_rate': 0.01,
                'subsample': 0.72,
                'subsample_freq': 4,
                'feature_fraction': 0.4,
                'lambda_l1': 1,
                'lambda_l2': 1,
                'seed': 46,
            }

            lgbm_model = LGBMRegressor(**params)
            lgbm_model.fit(df.drop(["Asset_ID", "y"], axis=1), df["y"])
            return lgbm_model

        for asset_id, asset_name in zip(self.asset_details['Asset_ID'], self.asset_details['Asset_Name']):
            start_time = time.time()
            print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
            lgbm_model = get_model_for_asset(df, y, asset_id)
            print(f"...took {round(time.time() - start_time, 2)} seconds")
            self._models[asset_id] = lgbm_model

    def predict(self, df):
        # TODO: alert when predicting unfitted
        #def predict_on_row(row):

        df["Prediction"] = np.nan
        #return df.apply(predict_on_row)

        prediction_features = OUTPUT_FEATURES.copy()
        prediction_features.remove("Asset_ID")
        for asset_id in self.asset_details.Asset_ID.values:
            df.loc[df["Asset_ID"] == asset_id, "Prediction"] = self._models[asset_id].predict(
                df[df["Asset_ID"] == asset_id][prediction_features]
            )

        return df["Prediction"].values

model = Model()
