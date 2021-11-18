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
        df_test = df_train[df_train.test_time_window == fold].reset_index(drop=True)
        train_until_time = df_test.datetime.min()
        df_train = df_train[df_train.datetime < train_until_time].reset_index(drop=True)
        #df_test = pd.read_csv(VALIDATION_DATA_PATH)  # ust this instead to run on validation data

        y_train = df_train.Target.fillna(0).values
        y_test = df_test.Target.fillna(0).values

        asset_weight = asset_details[asset_details["Asset_ID"] == asset_id]["Weight"].values[0]
        weights_train = y_train * 0 + asset_weight
        weights_test = y_test * 0 + asset_weight

        pipeline = feature_engineering.FeaturePipeline()
        pipeline.fit()  # just passing
        df_train = pipeline.transform(df_train)
        df_test = pipeline.transform(df_test)

        model.fit_for_asset_id(df_train, y_train, asset_id, df_test, y_test)

        pred_train = model._models[asset_id].predict(df_train.drop(["Asset_ID"], axis=1))
        pred_test = model._models[asset_id].predict(df_test.drop(["Asset_ID"], axis=1))

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
