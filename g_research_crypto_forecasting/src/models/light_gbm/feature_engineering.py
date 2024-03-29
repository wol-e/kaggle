import numpy as np
import pandas as pd

import sys

sys.path.append("...")  # TODO: to find the config and other sources, is there a nicer way?

from config import ASSET_DETAILS_PATH, MARKET_DATA_PATH

from helpers.memory import reduce_memory_usage

TRAINING_FEATURES = ["timestamp", "Target", "Asset_ID", "Count", "Open", "High", "Low", "Close", "Volume", "VWAP"]

lags = [30, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

OUTPUT_FEATURES = ["Asset_ID", "Count", "Open", "High", "Low", "Close",
                   "Volume", "VWAP", "Upper_Shadow", "Lower_Shadow",
                   "close2open",
                   "market_close2market_open",
                   "market_ratio",
                   "high2low",
                   "mean_price",
                   "high2mean",
                   "low2mean",
                   #"high2median",
                   #"low2median",
                   "volume2count",
                   "Lag_Calc_Target",
                   #"Market_Open",
                   #"Market_High",
                   #"Market_Low",
                   #"Market_Close",
                   ] + [f"close2lag_close_{lag}" for lag in lags] + [f"lag_market_ratio_{lag}"for lag in lags]


class FeaturePipeline():
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, df):
        market = pd.read_csv(MARKET_DATA_PATH, index_col="timestamp")
        df = df[TRAINING_FEATURES].join(market, on="timestamp", how="left")

        df = reduce_memory_usage(df)

        # adding row based features inspired by https://www.kaggle.com/code1110/gresearch-simple-lgb-starter
        pd.options.mode.chained_assignment = None  # default='warn'
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / df["High"]
        df['Lower_Shadow'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / df["Low"]
        df['close2open'] = df['Close'] / df['Open']
        df['market_close2market_open'] = df['Market_Close'] / df['Market_Open']
        df['market_ratio'] = df['close2open'] / df['market_close2market_open']
        df['high2low'] = df['High'] / df['Low']
        df['mean_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        #median_price = df[['Open', 'High', 'Low', 'Close']].median(axis=1)
        df['high2mean'] = df['High'] / df['mean_price']
        df['low2mean'] = df['Low'] / df['mean_price']
        #df['high2median'] = df['High'] / median_price
        #df['low2median'] = df['Low'] / median_price
        df['volume2count'] = df['Volume'] / (df['Count'] + 1)


        # adding target from 15 min ago (we manually recalculate the target in order
        # to be able to do this as well for the hidden test set)
        def calculate_target(data: pd.DataFrame, details: pd.DataFrame):
            """
            https://www.kaggle.com/alexfir/recreating-target/notebook
            """
            ids = list(details.Asset_ID)
            asset_names = list(details.Asset_Name)
            weights = np.array(list(details.Weight))
            price_column = "Close"

            times = data['timestamp'].agg(['min', 'max']).to_dict()
            all_timestamps = np.arange(times['min'], times['max'] + 60, 60)
            targets = pd.DataFrame(index=all_timestamps)

            for i, id in enumerate(ids):
                asset = data[data.Asset_ID == id].set_index(keys='timestamp')
                price = pd.Series(index=all_timestamps, data=asset[price_column])
                targets[ids[i]] = np.log(
                    price.shift(periods=-16) /
                    price.shift(periods=-1)
                )

            targets['m'] = np.average(targets.fillna(0), axis=1, weights=weights)

            m = targets['m']

            num = targets.multiply(m.values, axis=0).rolling(3750).mean().values
            denom = m.multiply(m.values, axis=0).rolling(3750).mean().values
            beta = np.nan_to_num(num.T / denom, nan=0., posinf=0., neginf=0.)

            targets = targets - (beta * m.values).T
            targets.drop('m', axis=1, inplace=True)

            # change to long format
            targets = pd.DataFrame(targets.stack().reset_index())
            targets.columns = ["timestamp", "Asset_ID", "Lag_Calc_Target"]

            return targets

        def join_lag_targets(df, calc_targets, lag_minutes):
            calc_targets["timestamp"] += 60 * lag_minutes
            df = df.join(calc_targets.set_index(["timestamp", "Asset_ID"]), on=["timestamp", "Asset_ID"], how="left")
            return df

        def join_lag_feature(df, feature_name, lag_minutes):
            lags = df[["timestamp", "Asset_ID", feature_name]].copy()
            lags.columns = ["timestamp", "Asset_ID", f"lag_{feature_name}_{lag_minutes}"]
            lags["timestamp"] += 60 * lag_minutes

            return df.join(lags.set_index(["timestamp", "Asset_ID"]), on=["timestamp", "Asset_ID"], how="left")


        asset_details = pd.read_csv(ASSET_DETAILS_PATH)
        targets = calculate_target(df, asset_details)
        df = join_lag_targets(df, targets, 15)

        # add lag features
        for lag in lags:
            df = join_lag_feature(df, "Close", lag)
            df[f"close2lag_close_{lag}"] = df["Close"] / df[f"lag_Close_{lag}"]

            df = join_lag_feature(df, "market_ratio", lag)

        return df[OUTPUT_FEATURES]


def nan_imputation(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0)
