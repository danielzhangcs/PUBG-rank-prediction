# 0.0238 becomes 0.0202 after adjusted https://www.kaggle.com/wxxiao33/lightgbm
import numpy as np
import pandas as pd
from sklearn import preprocessing

"""
changed from https://www.kaggle.com/anycode/simple-nn-baseline

to run this kernel, pip install ultimate first from your custom packages
"""
import gc, sys

gc.enable()

from timeit import default_timer


class Timer(object):
    """ A timer as a context manager

    Wraps around a timer. A custom timer can be passed

    to the constructor. The default timer is timeit.default_timer.

    Note that the latter measures wall clock time, not CPU time!

    On Unix systems, it corresponds to time.time.

    On Windows systems, it corresponds to time.clock.



    Adapted from: https://github.com/brouberol/contexttimer/blob/master/contexttimer/__init__.py



    Keyword arguments:

        output -- if True, print output after exiting context.

                  if callable, pass output to callable.

        format -- str.format string to be used for output; default "took {} seconds"

        prefix -- string to prepend (plus a space) to output

                  For convenience, if you only specify this, output defaults to True.

    """

    def __init__(self, prefix="", timer=default_timer,

                 output=None, fmt="took {:.2f} seconds"):

        self.timer = timer

        self.output = output

        self.fmt = fmt

        self.prefix = prefix

        self.end = None

    def __call__(self):

        """ Return the current time """

        return self.timer()

    def __enter__(self):

        """ Set the start time """

        self.start = self()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):

        """ Set the end time """

        self.end = self()

        if self.prefix and self.output is None:
            self.output = True

        if self.output:

            output = " ".join([self.prefix, self.fmt.format(self.elapsed)])

            if callable(self.output):

                self.output(output)

            else:

                print(output)

        gc.collect()

    def __str__(self):

        return '%.3f' % (self.elapsed)

    @property
    def elapsed(self):

        """ Return the current elapsed time since start

        If the `elapsed` property is called in the context manager scope,

        the elapsed time bewteen start and property access is returned.

        However, if it is accessed outside of the context manager scope,

        it returns the elapsed time bewteen entering and exiting the scope.

        The `elapsed` property can thus be accessed at different points within

        the context manager scope, to time different parts of the block.

        """

        if self.end is None:

            # if elapsed is called in the context manager scope

            return self() - self.start

        else:

            # if elapsed is called out of the context manager scope

            return self.end - self.start


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(

        100 * (start_mem - end_mem) / start_mem))

    return df


def feature_engineering(input_dir="data/", is_train=True):
    # Credit: 1. https://www.kaggle.com/anycode/simple-nn-baseline-4

    #         2. https://www.kaggle.com/harshitsheoran/mlp-and-fe

    # When this function is used for the training data, load train_V2.csv :

    if is_train:

        print("processing train_V2.csv")

        df = pd.read_csv(input_dir + 'train_V2.csv')

        # Only take the samples with matches that have more than 1 player

        # there are matches with no players or just one player ( those samples could affect our model badly)

        df = df[df['maxPlace'] > 1]

    # When this function is used for the test data, load test_V2.csv :

    else:

        print("processing test_V2.csv")

        df = pd.read_csv(input_dir + 'test_V2.csv')

    df = reduce_mem_usage(df)

    gc.collect()

    # Make a new feature indecating the total distance a player cut :



    df['totalDistance'] = 0.038 * df['rideDistance'] + 0.96 * df["walkDistance"] + 0.002 * df["swimDistance"]

    df['headshotrate'] = df['kills'] / df['headshotKills']

    df['killStreakrate'] = df['killStreaks'] / df['kills']

    df['healthitems'] = df['heals'] + df['boosts']

    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']

    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']

    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']

    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']

    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']

    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']

    df["skill"] = df["headshotKills"] + df["roadKills"]

    df['rankPoints'] = np.where(df['rankPoints'] <= 0,

                                (df.killPoints + df.winPoints) / 2, df.rankPoints)

    df['killPoints'] = np.where(df['killPoints'] <= 0,

                                df.rankPoints, df.killPoints)

    df['winPoints'] = np.where(df['winPoints'] <= 0,

                               df.rankPoints, df.winPoints)

    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN

    df.fillna(0, inplace=True)

    # Process the 'rankPoints' feature by replacing any value of (-1) to be (0) :

    df.loc[df.rankPoints < 0, 'rankPoints'] = 0

    target = 'winPlacePerc'

    # Get a list of the features to be used

    features = df.columns.tolist()

    # Remove some features from the features list :

    features.remove("Id")

    features.remove("matchId")

    features.remove("groupId")

    features.remove("matchDuration")

    features.remove("matchType")

    features.remove("maxPlace")

    y = None

    # If we are processing the training data, process the target

    # (group the data by the match and the group then take the mean of the target)

    if is_train:

        with Timer("Calculating y:"):

            y = df.groupby(['matchId', 'groupId'])[

                target].first().values

            # Remove the target from the features list :

            features.remove(target)

    else:

        df_idx = df[["Id", "matchId", "groupId"]].copy()

    with Timer("Match level feature"):

        df_out = df.groupby(['matchId', 'groupId'])[

            ["maxPlace", "matchDuration"]].first().reset_index()

    df = df[features + ["matchId", "groupId"]].copy()

    gc.collect()

    with Timer("Mean features:"):

        # Make new features indicating the mean of the features ( grouped by match and group ) :

        agg = df.groupby(['matchId', 'groupId'])[

            features].agg('mean')

        # Put the new features into a rank form ( max value will have the highest rank)

        agg_rank = agg.groupby('matchId')[features].rank(

            pct=True).reset_index()

        agg_mean = agg.reset_index().groupby(

            'matchId')[features].mean()

        agg_mean.columns = [x + "_mean_mean" for x in agg_mean.columns]

    with Timer("Merging (mean):"):

        # Merge agg and agg_rank (that we got before) with df_out :

        df_out = df_out.merge(

            agg.reset_index(), how='left', on=['matchId', 'groupId'])

        df_out = df_out.merge(

            agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

        df_out = df_out.merge(

            agg_mean.reset_index(), how='left', on=['matchId'])

        df_out = reduce_mem_usage(df_out)

    with Timer("Max features:"):

        # Make new features indicating the max value of the features for each group ( grouped by match )

        agg = df.groupby(['matchId', 'groupId'])[features].agg('max')

        # Put the new features into a rank form ( max value will have the highest rank)

        agg_rank = agg.groupby('matchId')[features].rank(

            pct=True).reset_index()

        agg_mean = agg.groupby('matchId')[features].mean()

        agg_mean.columns = [x + "_max_mean" for x in agg_mean.columns]

    with Timer("Merging (max):"):

        # Merge the new (agg and agg_rank) with df_out :

        df_out = df_out.merge(

            agg.reset_index(), how='left', on=['matchId', 'groupId'])

        df_out = df_out.merge(agg_rank, suffixes=[

            "_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

        df_out = df_out.merge(

            agg_mean.reset_index(), how='left', on=['matchId'])

        df_out = reduce_mem_usage(df_out)

    with Timer("Min features:"):

        # Make new features indicating the minimum value of the features for each group ( grouped by match )

        agg = df.groupby(['matchId', 'groupId'])[features].agg('min')

        # Put the new features into a rank form ( max value will have the highest rank)

        agg_rank = agg.groupby('matchId')[features].rank(

            pct=True).reset_index()

    with Timer("Merging (min):"):

        # Merge the new (agg and agg_rank) with df_out :

        df_out = df_out.merge(agg.reset_index(), how='left', on=[

            'matchId', 'groupId'])

        df_out = df_out.merge(agg_rank, suffixes=[

            "_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

        df_out = reduce_mem_usage(df_out)

    with Timer("Sum features:"):

        # Make new features indicating the minimum value of the features for each group ( grouped by match )

        agg = df.groupby(['matchId', 'groupId'])[features].agg('sum')

        # Put the new features into a rank form ( max value will have the highest rank)

        agg_rank = agg.groupby('matchId')[features].rank(

            pct=True).reset_index()

    with Timer("Merging (sum):"):

        # Merge the new (agg and agg_rank) with df_out :

        df_out = df_out.merge(agg.reset_index(), how='left', on=[

            'matchId', 'groupId'])

        df_out = df_out.merge(agg_rank, suffixes=[

            "_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])

        df_out = reduce_mem_usage(df_out)

    # Make new features indicating the number of players in each group ( grouped by match )

    with Timer("Group size:"):

        agg = df.groupby(['matchId', 'groupId']).size(

        ).reset_index(name='group_size')

        # Merge the group_size feature with df_out :

        df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    with Timer("Match mean feature"):

        # Make new features indicating the mean value of each features for each match :

        agg = df.groupby(['matchId'])[features].agg('mean').reset_index()

        # Merge the new agg with df_out :

        df_out = df_out.merge(

            agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

        df_out = reduce_mem_usage(df_out)

    with Timer("Match median feature"):

        # Make new features indicating the mean value of each features for each match :

        agg = df.groupby(['matchId'])[features].agg('median').reset_index()

        # Merge the new agg with df_out :

        df_out = df_out.merge(

            agg, suffixes=["", "_match_median"], how='left', on=['matchId'])

        df_out = reduce_mem_usage(df_out)

    with Timer("Match size feature"):

        # Make new features indicating the number of groups in each match :

        agg = df.groupby(['matchId']).size().reset_index(name='match_size')

        # Merge the match_size feature with df_out :

        df_out = df_out.merge(agg, how='left', on=['matchId'])

    df_out[df_out == np.Inf] = np.NaN

    df_out[df_out == np.NINF] = np.NaN

    df_out.fillna(0, inplace=True)

    df_out = reduce_mem_usage(df_out)

    gc.collect()

    if is_train:
        # Drop matchId and groupId
        df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
        return df_out, y

    return df_out, df_idx


x_train, y_train = feature_engineering("../input/", is_train=True)

gc.collect()

x_test, df_sub = feature_engineering("../input/", is_train=False)
df_test_cleaned = x_test[["matchId", "groupId", "maxPlace", "numGroups"]].copy()
x_test.drop(["matchId", "groupId"], axis=1, inplace=True)
gc.collect()

import os
import time
import gc
import warnings

warnings.filterwarnings("ignore")
# data manipulation

import numpy as np
import pandas as pd
# plot
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# excluded_features = []
# use_cols = [col for col in df_train.columns if col not in excluded_features]

train_index = round(int(x_train.shape[0] * 0.8))
dev_X = x_train[:train_index]
val_X = x_train[train_index:]
dev_y = y_train[:train_index]
val_y = y_train[train_index:]
gc.collect();


# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, x_test):
    params = {"objective": "regression", "metric": "mae", 'n_estimators': 30000, 'early_stopping_rounds': 200,
              "num_leaves": 63, "learning_rate": 0.03, "bagging_fraction": 0.7,
              "bagging_seed": 0, "num_threads": 4, "colsample_bytree": 0.7,
              }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)

    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_test_y, model


# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)

df_test_cleaned["winPlacePerc"] = pred_test

df_test_cleaned.loc[

    df_test_cleaned.winPlacePerc < 0, "winPlacePerc"] = 0

df_test_cleaned.loc[

    df_test_cleaned.winPlacePerc > 1, "winPlacePerc"] = 1

df_sub = df_sub.merge(

    df_test_cleaned[["matchId", "groupId", "winPlacePerc"]], how="left",

    on=["matchId", "groupId"]

)

df_sub[['Id', 'winPlacePerc']].to_csv("submission_raw.csv", index=False)
