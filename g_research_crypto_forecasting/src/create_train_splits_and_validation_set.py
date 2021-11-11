import pandas as pd
from sklearn import model_selection
from config import NUMBER_FOLDS

df = pd.read_csv("../data/raw/train.csv")

df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df_valid = df[df['datetime'] >= '2021-06-13 00:00:00']  # according to public opinion on kaggle, this is where the public test set starts to leak in
df = df[df['datetime'] < '2021-06-13 00:00:00']

# TODO: this is not a good wat to create folds for a timeseries problem, better do timewise splits
df["fold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
folds = model_selection.KFold(n_splits=NUMBER_FOLDS)

for i, (train, test) in enumerate(folds.split(X=df, y=df.Target)):
    df.loc[test, "fold"] = i

df.to_csv("../data/processed/train_folds.csv", index=False)
df_valid.to_csv("../data/processed/valid.csv", index=False)