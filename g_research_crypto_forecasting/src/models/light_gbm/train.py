import joblib
import numpy as np
import pandas as pd
import time
from sklearn import metrics

import sys

sys.path.append("...")  # TODO: to find the config. there must be a better way.

from config import TRAINING_DATA_PATH, ASSET_DETAILS_PATH, VALIDATION_DATA_PATH
from custom_metrics import custom_metrics

from . import feature_engineering
from .model import Model


def run(fold, save_model=False, validate=False):
    """
    :param fold:
    :param save_model:
    :param validate:
    :return:
    """
    print(f"\ntraining on fold {fold}...\n")

    asset_details = pd.read_csv(ASSET_DETAILS_PATH)
    asset_ids = asset_details["Asset_ID"].values
    asset_names = asset_details["Asset_Name"].values
    asset_details = asset_details.set_index("Asset_ID", drop=True)

    model = Model()

    weights_train_full = np.array([])
    weights_test_full = np.array([])
    pred_train_full = np.array([])
    pred_test_full = np.array([])
    y_test_full = np.array([])
    y_train_full = np.array([])

    for i, asset_id in enumerate(asset_ids):  # we do it like this for better memory efficiency
        df_train = pd.read_csv(TRAINING_DATA_PATH[:-4] + f"_asset_id_{asset_id}.csv")

        # test set is identified by 'test_time_window' value, train set is everything >before< that time window
        #df_test = df_train[df_train.test_time_window == fold].reset_index(drop=True)
        #train_until_time = df_test.datetime.min()
        #df_train = df_train[df_train.datetime < train_until_time].reset_index(drop=True)
        df_test = pd.read_csv(VALIDATION_DATA_PATH)  # ust this instead to run on validation data

        y_train = df_train.Target.fillna(0).values
        y_test = df_test.Target.fillna(0).values

        weights_train = df_train[["Asset_ID"]].copy()
        weights_test = df_test[["Asset_ID"]].copy()
        weights_train = weights_train.join(asset_details, how="left", on="Asset_ID")["Weight"].values
        weights_test = weights_test.join(asset_details, how="left", on="Asset_ID")["Weight"].values

        pipeline = feature_engineering.FeaturePipeline()
        pipeline.fit()  # just passing
        df_train = pipeline.transform(df_train)
        df_test = pipeline.transform(df_test)

        train_start = time.time()
        model.fit_for_asset_id(df_train, y_train, asset_id)

        pred_train = model._models[asset_id].predict(df_train.drop(["Asset_ID"], axis=1))
        pred_test = model._models[asset_id].predict(df_test.drop(["Asset_ID"], axis=1))

        rmse_test = metrics.mean_squared_error(y_true=y_test, y_pred=pred_test, squared=False)
        rmse_train = metrics.mean_squared_error(y_true=y_train, y_pred=pred_train, squared=False)
        corr_test = custom_metrics.weighted_correlation_coefficient(y_true=y_test, y_pred=pred_test, weights=weights_test)
        corr_train = custom_metrics.weighted_correlation_coefficient(y_true=y_train, y_pred=pred_train,
                                                                     weights=weights_train)

        print(f"""Fold {fold} for {asset_names[i]}:
    
        RMSE Train: {rmse_train}, Test: {rmse_test}
        WEIGHTED CORR Train: {corr_train}, Test: {corr_test}
        training and scoring time: {round(time.time() - train_start, 2)} s
    
        """)  # TODO: add some meaningful metric for the absolute diff. rmspe would be good, but has problems with data centered around 0 (zero division errors)

        weights_train_full = np.concatenate((weights_train_full, weights_train))
        weights_test_full = np.concatenate((weights_test_full, weights_test))
        pred_train_full = np.concatenate((pred_train_full, pred_train))
        pred_test_full = np.concatenate((pred_test_full, pred_test))
        y_test_full = np.concatenate((y_test_full, y_test))
        y_train_full = np.concatenate((y_train_full, y_train))

    corr_test_full = custom_metrics.weighted_correlation_coefficient(
        y_true=y_test_full,
        y_pred=pred_test_full,
        weights=weights_test_full
    )
    corr_train_full = custom_metrics.weighted_correlation_coefficient(
        y_true=y_train_full,
        y_pred=pred_train_full,
        weights=weights_train_full
    )
    print(f"""Fold {fold} Total:

    WEIGHTED CORR Train: {corr_train_full}, Test: {corr_test_full}

    """)

    if save_model:
        joblib.dump(model, f"saved_models/light_gbm/light_gbm_fold_{fold}.joblib")
