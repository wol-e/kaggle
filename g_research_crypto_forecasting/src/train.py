import argparse
import joblib
import importlib
import pandas as pd
import time
from sklearn import metrics

from config import TRAINING_DATA_PATH

def run(model_name, fold, save_model=False):
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

    feature_engineering = importlib.import_module(f"models.{model_name}.feature_engineering")
    model = importlib.import_module(f"models.{model_name}.model").model

    pipeline = feature_engineering.FeaturePipeline()
    pipeline.fit(pd.concat([df_train, df_test]))
    df_train, df_test = pipeline.transform(df_train), pipeline.transform(df_test)

    train_start = time.time()
    model.fit(df_train, y_train)

    #TODO: rmse is a bad metric for this, better use e.g. rmspe or the actual competition metric
    rmse_test = metrics.mean_squared_error(y_true=y_test, y_pred=model.predict(df_test), squared=False)
    rmse_train = metrics.mean_squared_error(y_true=y_train, y_pred=model.predict(df_train), squared=False)
    print(f"RMSE on fold {fold}: Train: {rmse_train}, Test: {rmse_test}, training and scoring time: {round(time.time() - train_start,2)} s")

    if save_model:
        joblib.dump(model, f"saved_models/{model_name}/{model_name}_fold_{fold}.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    run(model_name=args.model, fold=args.fold)