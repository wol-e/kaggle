import pandas as pd
import numpy as np
from .feature_engineering import OUTPUT_FEATURES
from lightgbm import LGBMRegressor
import sys
import time
from custom_metrics import custom_metrics
from sklearn import metrics

sys.path.append("...")  # TODO: to find the config. there must be a better way.

from config import ASSET_DETAILS_PATH, TRAINING_DATA_PATH


class Model:
    def __init__(self):
        self._models = {}
        self.asset_details = pd.read_csv(ASSET_DETAILS_PATH).sort_values("Asset_ID")


    def fit(self, df, y):
        for asset_id in self.asset_details['Asset_ID']:
            self.fit_for_asset_id(df, y, asset_id)

    def fit_for_asset_id(self, df, y, asset_id, df_test=None, y_test=None):
        df = df.copy()
        #df = df.dropna(how="any")

        df["y"] = y
        df = df[df["Asset_ID"] == asset_id]
        y = df["y"]

        params = {  # TODO: param search
            'n_estimators': 1000,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': 10,
            'learning_rate': 0.01,
            'subsample': 0.72,
            'subsample_freq': 4,
            #'feature_fraction': 0.4,
            'lambda_l1': 10,
            #'lambda_l2': 1,
            'seed': 99  ,
            'verbose': -1,
        }

        start_time = time.time()
        asset_name = self.asset_details[self.asset_details["Asset_ID"] == asset_id]["Asset_Name"].values[0]
        print(f"\nTraining model for {asset_name:<16} (ID={asset_id:<2})")
        lgbm_model = LGBMRegressor(**params)
        lgbm_model.fit(df.drop(["Asset_ID", "y"], axis=1), df["y"])
        self._models[asset_id] = lgbm_model
        scores = ""
        if df_test is not None:
            df_test = df_test.copy()
            df_test["y"] = y_test
            df_test = df_test[df_test["Asset_ID"] == asset_id]
            y_test = df_test["y"]

            asset_weight = self.asset_details[self.asset_details["Asset_ID"] == asset_id]["Weight"].values[0]
            pred_train = self._models[asset_id].predict(df.drop(["Asset_ID", "y"], axis=1))
            pred_test = self._models[asset_id].predict(df_test.drop(["Asset_ID", "y"], axis=1))

            rmse_test = round(metrics.mean_squared_error(y_true=y_test, y_pred=pred_test, squared=False), 5)
            rmse_train = round(metrics.mean_squared_error(y_true=y, y_pred=pred_train, squared=False), 5)
            corr_test = round(custom_metrics.weighted_correlation_coefficient(y_true=y_test, y_pred=pred_test,
                                                                        weights=y_test * 0 + asset_weight), 5)
            corr_train = round(custom_metrics.weighted_correlation_coefficient(y_true=y, y_pred=pred_train,
                                                                         weights=y * 0 + asset_weight), 5)

            scores = f"\tRMSE: (Train {rmse_train}, Test {rmse_test}) WEIGHTED CORR: (Train {corr_train}, Test {corr_test})"
        print(f"...took {round(time.time() - start_time, 2)} seconds{scores}")


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

