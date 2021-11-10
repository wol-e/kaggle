import argparse
import joblib
import importlib
import pandas as pd

from config import TRAINING_DATA_PATH, TEST_DATA_PATH, NUMBER_FOLDS, SAMPLE_SUBMISSION_PATH

def run(model_name):
    """

    :param mondel_name str:
    :return: None
    """
    df_train = pd.read_csv(TRAINING_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    feature_engineering = importlib.import_module(f"models.{model_name}.feature_engineering")

    pipeline = feature_engineering.FeaturePipeline()
    pipeline.fit(df_train)
    df_test = pipeline.transform(df_test)

    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission["target"] = 0

    for fold in range(NUMBER_FOLDS):
        model = joblib.load(f"saved_models/{model_name}/{model_name}_fold_{fold}.joblib")
        predictions = model.predict_proba(df_test)[:, 1]
        submission["target"] += predictions

    submission["target"] /= NUMBER_FOLDS
    submission.to_csv(f"submissions/{model_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    run(model_name=args.model)