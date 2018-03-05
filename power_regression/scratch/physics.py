import os
import glob

import pandas as pd
import numpy as np

from joblib import Parallel, delayed, Memory

from skcycling.io import bikeread
from skcycling.extraction import acceleration
from skcycling.extraction import gradient_elevation
from skcycling.model import strava_power_model

path_data = '/home/glemaitre/Documents/data/cycling/user_*/*/*.fit'
filenames = sorted(glob.glob(path_data))
memory = Memory(location='../notebook/bikereadcache')
bikeread_cached = memory.cache(bikeread, verbose=1)
data = Parallel(n_jobs=-1)(delayed(bikeread_cached)(f) for f in filenames)

# filter the activity which do not contain the required information
fields = ['elevation', 'cadence', 'distance', 'heart-rate', 'power', 'speed']
valid_data = []
valid_filenames = []
for activity, filename in zip(data, filenames):
    if set(fields).issubset(activity.columns):
        if not pd.isnull(activity).any().any():
            valid_data.append(activity)
            valid_filenames.append(filename)
data = valid_data

for activity_idx in range(len(data)):
    # compute acceleration
    data[activity_idx] = acceleration(data[activity_idx])
    # compute gradient elevation
    data[activity_idx] = gradient_elevation(data[activity_idx])

for activity in data:
    activity.replace([np.inf, -np.inf], np.nan, inplace=True)

data_concat = pd.concat(data)
y = data_concat['power']
X = data_concat.drop('power', axis=1)
X.fillna(X.mean(), inplace=True)
groups = []
for group_idx, activity in enumerate(data):
    groups += [group_idx] * activity.shape[0]
groups = np.array(groups)

weight_user = {'user_1': 86., 'user_2': 72., 'user_3': 61., 'user_5': 72.}
weight_groups = np.zeros_like(groups)
for idx, filename in enumerate(valid_filenames):
    mask = groups == idx
    weight = [value for key, value in weight_user.items() if key in filename]
    weight_groups[mask] = weight[0]

y_pred = np.zeros_like(y.values)
for idx_activity in range(len(valid_filenames)):
    mask = groups == idx
    y_pred[mask] = strava_power_model(X.iloc[mask, :],
                                      np.unique(weight_groups[mask])).values

    path_results = ('/home/glemaitre/Documents/work/code/cycling/research/'
                'power_regression/results')
np.save(os.path.join(path_results, 'y_pred_physics.npy'), y_pred)
np.save(os.path.join(path_results, 'y_true_physics.npy'), y.values)
