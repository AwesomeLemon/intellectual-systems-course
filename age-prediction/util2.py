import csv
import pandas as pd
import gc
from scipy.sparse import coo_matrix, csr_matrix

import numpy as np
import read_data
import os

def add_column(primary_file_path, column_file_path):
    data = np.array(pd.read_csv(primary_file_path, delimiter=",", header=None).as_matrix())
    column = np.array(pd.read_csv(column_file_path, delimiter=",", header=None).as_matrix())
    new_data = np.c_[data[:, :4], column[:, 1], data[:, 4]]
    np.savetxt('smth.txt', new_data, delimiter=",")

add_column(read_data.extracted_features_train_path, read_data.additional_features_train_path)