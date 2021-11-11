import argparse
import joblib
import importlib
import pandas as pd
import time
from sklearn import metrics

from config import TRAINING_DATA_PATH, ASSET_DETAILS_PATH, VALIDATION_DATA_PATH
from custom_metrics import custom_metrics

def run(model_name, fold, save_model=False, validate=False):
    """
    runs training on provided fold, i.e. uses the training data with matching fold as test data and trains on remaining
    data.

    :param fold int: fold number to run training for
    :return: None
    """
    df_train = pd.read_csv(TRAINING_DATA_PATH)

    df_test = df_train[df_train.fold == fold].reset_index(drop=True)
    df_train = df_train[df_train.fold != fold].reset_index(drop=True)

    y_train = df_train.Target.fillna(0).values
    y_test = df_test.Target.fillna(0).values

    weights_train = df_train[["Asset_ID"]].copy()
    weights_test = df_test[["Asset_ID"]].copy()
    asset_details = pd.read_csv(ASSET_DETAILS_PATH, index_col="Asset_ID")
    weights_train = weights_train.join(asset_details, how="left", on="Asset_ID")["Weight"].values
    weights_test = weights_test.join(asset_details, how="left", on="Asset_ID")["Weight"].values

    feature_engineering = importlib.import_module(f"models.{model_name}.feature_engineering")
    model = importlib.import_module(f"models.{model_name}.model").model

    pipeline = feature_engineering.FeaturePipeline()
    pipeline.fit(pd.concat([df_train, df_test]))
    df_train, df_test = pipeline.transform(df_train), pipeline.transform(df_test)

    train_start = time.time()
    model.fit(df_train, y_train)

    pred_train = model.predict(df_train)
    pred_test = model.predict(df_test)

    rmse_test = metrics.mean_squared_error(y_true=y_test, y_pred=pred_test, squared=False)
    rmse_train = metrics.mean_squared_error(y_true=y_train, y_pred=pred_train, squared=False)
    rmspe_test = custom_metrics.rmspe(y_true=y_test, y_pred=pred_test, weights=None)  # TODO: zero division
    rmspe_train = custom_metrics.rmspe(y_true=y_train, y_pred=pred_train, weights=None)
    corr_test = custom_metrics.weighted_correlation_coefficient(y_true=y_test, y_pred=pred_test, weights=weights_test)
    corr_train = custom_metrics.weighted_correlation_coefficient(y_true=y_train, y_pred=pred_train, weights=weights_train)

    print(f"""Fold {fold}:
    
    RMSE Train: {rmse_train}, Test: {rmse_test}
    RMSPE Train: {rmspe_train}, Test: {rmspe_test}
    WEIGHTED CORR Train: {corr_train}, Test: {corr_test}
    training and scoring time: {round(time.time() - train_start,2)} s
        
    """)

    if validate:
        df_valid = pd.read_csv(VALIDATION_DATA_PATH)
        y_valid = df_valid.Target.fillna(0).values
        weights_valid = df_valid[["Asset_ID"]].copy()
        weights_valid = weights_valid.join(asset_details, how="left", on="Asset_ID")["Weight"].values
        df_valid = pipeline.transform(df_valid)
        pred_valid = model.predict(df_valid)
        corr_valid = custom_metrics.weighted_correlation_coefficient(y_true=y_valid, y_pred=pred_valid,
                                                                     weights=weights_valid)
        print(f"""Fold {fold}:
        
        WEIGHTED CORR Valid: {corr_valid}
        """)

    if save_model:
        joblib.dump(model, f"saved_models/{model_name}/{model_name}_fold_{fold}.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    run(model_name=args.model, fold=args.fold)