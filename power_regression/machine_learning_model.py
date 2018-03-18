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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

from xgboost import XGBRegressor as GradientBoostingRegressor

###############################################################################
# IO section:
# 1. Read all available data
# 2. Remove the files which do not contain all required information

# cache the reading if we need to execute the script again
memory = Memory(location='../cache/bikereadcache')
bikeread_cached = memory.cache(bikeread, verbose=1)

# read the data
path_data = '/home/glemaitre/Documents/data/user_*/*/*.fit'
filenames = sorted(glob.glob(path_data))
data = Parallel(n_jobs=-1)(delayed(bikeread_cached)(f) for f in filenames)

# filter the activity which do not contain the required information
fields = ['elevation', 'cadence', 'distance', 'heart-rate', 'power', 'speed']
valid_data = []
for activity in data:
    if set(fields).issubset(activity.columns):
        if not pd.isnull(activity).any().any():
            valid_data.append(activity)
data = valid_data

###############################################################################
# Data processing
# 1. Compute extra information: acceleration, gradient for elevation and
# heart-rate. and compute gradient over 5 seconds.
# 2. Use a standard scaler and a gradient boosting as estimator.
# 3. Make a cross-validation to obtain true estimate of the score.
# 4. Repeat the experiment to get the prediction for visualization purpose.

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

print('The obtained scores on training and testing in terms of '
      'R2 and MAE are: \n')
print(scores)

# Store the prediction for visualization
y_pred = cross_val_predict(pipe, X, y, groups=groups,
                           cv=GroupKFold(n_splits=3), n_jobs=1)
path_results = os.path.join('results', 'machine_learning_model')
f = os.path.join(path_results, 'y_pred.csv')
pd.Series(y_pred, index=y.index).to_csv(f)
f = os.path.join(path_results, 'y_true.csv')
y.to_csv(f)
np.save(os.path.join(path_results, 'groups.npy'), groups)
