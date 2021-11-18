import argparse
import joblib
import importlib
import pandas as pd
import time
from sklearn import metrics

from config import TRAINING_DATA_PATH, ASSET_DETAILS_PATH, VALIDATION_DATA_PATH
from custom_metrics import custom_metrics

def run(model_name, fold, save_model=False, validate=False):

    train = importlib.import_module(f"models.{model_name}.train")

    train.run(fold=fold, save_model=save_model, validate=validate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    run(model_name=args.model, fold=args.fold)