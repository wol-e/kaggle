import pandas as pd
from sklearn import model_selection
from config import NUMBER_FOLDS
from datetime import timedelta

df = pd.read_csv("../data/raw/train.csv")

df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df_valid = df[df['datetime'] >= '2021-06-13 00:00:00']  # according to public opinion on kaggle, this is where the public test set starts to leak in
df = df[df['datetime'] < '2021-06-13 00:00:00']

# add folds based on simple randomization (not good for time-series problems)
df["test_fold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
folds = model_selection.KFold(n_splits=NUMBER_FOLDS)

for i, (train, test) in enumerate(folds.split(X=df, y=df.Target)):
    df.loc[test, "test_fold"] = i

# add splits based on time-series slices
n_test_days = 90  # every test fold
max_train_datetime = df.datetime.max()
df["test_time_window"] = -1
for i in range(NUMBER_FOLDS):
    end_test_window = max_train_datetime - timedelta(days=i * n_test_days)
    start_test_window = max_train_datetime - timedelta(days=(i + 1) * n_test_days)
    test_filter = ((df["datetime"] <= end_test_window) & (df["datetime"] >= start_test_window))
    df.loc[test_filter, "test_time_window"] = i

df.to_csv("../data/processed/train_folds.csv", index=False)
df_valid.to_csv("../data/processed/valid.csv", index=False)