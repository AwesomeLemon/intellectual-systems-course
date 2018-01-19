import csv
import pandas as pd
import gc
from scipy.sparse import coo_matrix, csr_matrix
from util import *
import numpy as np
import os

def transform(path):
    data = np.array(pd.read_csv(path, delimiter=",", header=None).as_matrix())
    # print(data[:, -1].mean())
    val_to_subtract = 500
    data[:, 1] = data[:, 1] - val_to_subtract
    np.savetxt('subtract_' + str(val_to_subtract) + '.txt', data, fmt='%d', delimiter=',')
    # multiplyer = 2.0
    # data[:, 1] = data[:, 1] * multiplyer
    # np.savetxt('mult_by_' + str(multiplyer) + '.txt', data, fmt='%d', delimiter=',')

def use_parent_bdays(path):
    data = np.array(pd.read_csv(path, delimiter=",", header=None).as_matrix())
    parent_birthdates_dict = retrieve_obj('data/parents_birthdays')
    expected_year_diff = 23 # 26 gives best results
    expected_day_diff = expected_year_diff * 365
    for i, user_predictedbd in enumerate(data):
        if user_predictedbd[0] in parent_birthdates_dict:
            data[i, 1] = parent_birthdates_dict[user_predictedbd[0]] + expected_day_diff
    np.savetxt('pred_parents_'+path+str(expected_year_diff) +'.txt', data, fmt='%d', delimiter=',')

def average_with_parent_bdays(path):
    data = np.array(pd.read_csv(path, delimiter=",", header=None).as_matrix())
    parent_birthdates_dict = retrieve_obj('data/parents_birthdays')
    expected_year_diff = 27
    expected_day_diff = expected_year_diff * 365
    for i, user_predictedbd in enumerate(data):
        if user_predictedbd[0] in parent_birthdates_dict:
            data[i, 1] = 0.5 * (parent_birthdates_dict[user_predictedbd[0]] + expected_day_diff) + 0.5*user_predictedbd[1]
    np.savetxt('pred_parents_'+path+str(expected_year_diff) +'_avg0.5.txt', data, fmt='%d', delimiter=',')

def get_20_percent():
    data = np.array(pd.read_csv('data/user_features_test', delimiter=",", header=None).as_matrix())
    np.savetxt('minus20percent', np.c_[data[:, 0], data[:, 4] / data[:, 3]],  fmt='%d', delimiter=',')

# def addaveragesanddiffs(path):
#     data = np.array(pd.read_csv(path, delimiter=",", header=None).as_matrix())
#     np.savetxt('lotsofstuff.txt', np.c_[data, data[:], fmt='%d', delimiter=',')

average_with_parent_bdays('minus20percent')
# use_parent_bdays('pred_best.txt')
# transform('pred_best.txt')
# transform('data/user_features_train')
# get_20_percent()