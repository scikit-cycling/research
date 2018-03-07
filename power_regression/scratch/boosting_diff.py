import os
import glob

import pandas as pd
import numpy as np

from joblib import Parallel, delayed, Memory

from skcycling.io import bikeread
from skcycling.extraction import acceleration
from skcycling.extraction import gradient_activity
from skcycling.extraction import gradient_elevation
from skcycling.extraction import gradient_heart_rate

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor as GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

path_data = '/home/glemaitre/Documents/data/cycling/user_*/*/*.fit'
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
    # compute the gradient information over 10 sec for the some fields
    fields = ['elevation', 'cadence', 'heart-rate', 'speed',
              'gradient-elevation', 'gradient-heart-rate', 'acceleration']
    data[activity_idx] = gradient_activity(data[activity_idx],
                                           periods=range(1, 6),
                                           columns=fields)

for activity in data:
    activity.replace([np.inf, -np.inf], np.nan, inplace=True)

data_concat = pd.concat(data)
y = data_concat['original']['power']
X = data_concat.drop('power', axis=1, level=1)
X.fillna(X.mean(), inplace=True)
groups = []
for group_idx, activity in enumerate(data):
    groups += [group_idx] * activity.shape[0]
groups = np.array(groups)

pipe = make_pipeline(StandardScaler(),
                     GradientBoostingRegressor(random_state=42, n_jobs=-1))
scores = cross_validate(pipe,
                        X, y, groups=groups,
                        scoring=['r2', 'neg_median_absolute_error'],
                        cv=GroupKFold(n_splits=3), n_jobs=1,
                        return_train_score=True,
                        verbose=0)

print(scores)

y_pred = cross_val_predict(pipe, X, y, groups=groups,
                           cv=GroupKFold(n_splits=3), n_jobs=1)

path_results = ('/home/glemaitre/Documents/work/code/cycling/research/'
                'power_regression/results')
f = os.path.join(path_results, 'y_pred_boosting.csv')
pd.Series(y_pred, index=y.index).to_csv(f)
f = os.path.join(path_results, 'y_true_boosting.csv')
y.to_csv(f)
np.save(os.path.join(path_results, 'groups_boosting.npy'), groups)
