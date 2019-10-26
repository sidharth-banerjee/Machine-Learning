'''
Name: Sidharth Banerjee
ID  : 1001622703
'''

import sys
import math
import random
import numpy as np
import pandas as pd

# command line arguments
training_file = str(sys.argv[1])
test_file = str(sys.argv[2])
option = str(sys.argv[3])
pruning_thr= int(sys.argv[4])

# # Functions

# function to return csv data in a DataFrame format
def returnData(training_file):
    df = pd.read_csv(training_file, delimiter= '\s+', header=None)
    attributes = len(df.columns) - 1
    col_header = np.arange(1, attributes+1, 1)
    col_header = col_header.tolist()
    col_header.append('Class')
    df.columns = col_header
    return df

df_train = returnData(training_file)
print(df_train.head())
