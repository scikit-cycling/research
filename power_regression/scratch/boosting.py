import glob

import pandas as pd
import numpy as np

from joblib import Parallel, delayed, Memory

from skcycling.io import bikeread
from skcycling.extraction import acceleration
from skcycling.extraction import gradient_elevation
from skcycling.extraction import gradient_heart_rate

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate

path_data = '/home/lemaitre/Documents/data/cycling/user_2/*/*.fit'
filenames = sorted(glob.glob(path_data))
memory = Memory(location='../notebook/bikereadcache')
bikeread_cached = memory.cache(bikeread, verbose=1)
data = Parallel(n_jobs=-1)(delayed(bikeread_cached)(f) for f in filenames)

# filter the activity which do not contain the required information
fields = ['elevation', 'cadence', 'distance', 'heart-rate', 'power', 'speed']
valid_data = []
for activity in data:
    if set(fields).issubset(activity.columns):
        if not pd.isnull(activity).any().any():
            valid_data.append(activity)
data = valid_data

for activity_idx in range(len(data)):
    # compute acceleration
    data[activity_idx] = acceleration(data[activity_idx])
    # compute gradient elevation
    data[activity_idx] = gradient_elevation(data[activity_idx])
    # compute gradient heart-rate
    data[activity_idx] = gradient_heart_rate(data[activity_idx])

for activity in data:
    activity.replace([np.inf, -np.inf], np.nan, inplace=True)
    activity.fillna(activity.mean(), inplace=True)

data_concat = pd.concat(data)
y = data_concat['power']
X = data_concat.drop('power', axis=1)
groups = []
for group_idx, activity in enumerate(data):
    groups += [group_idx] * activity.shape[0]
groups = np.array(groups)

pipe = make_pipeline(QuantileTransformer(),
                     GradientBoostingRegressor(random_state=42))
scores = cross_validate(pipe, X, y, groups=groups,
                        scoring=['r2', 'neg_median_absolute_error'],
                        cv=GroupKFold(n_splits=3), n_jobs=-1,
                        return_train_score=True)
print(scores)
