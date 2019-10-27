'''
Name: Sidharth Banerjee
ID  : 1001622703
'''

import sys
import math
import random
import numpy as np
import pandas as pd

from node import Node

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

def class_distribution(df):
    class_count = [0] * (df['Class'].max())
    for i in range (0, len(df)):
        class_count[int(df.iloc[i]['Class'])-1] += 1
    class_probability = np.array(class_count)/1000
    return class_probability

def DTL_TopLevel(df, pruning_thr):
    attributes = len(df.columns)-1
    default = np.argmax(class_distribution(df)) + df['Class'].min()
    return (np.array(df), attributes, default, pruning_thr)

#def DTL(examples, attributes, default, pruning_thr):
#    if np.size(examples) == 0:
#        return default

#    elif(sameClass(examples)):
#        return examples[0][-1]

df_train = returnData(training_file)

arr, att, default, pru = DTL_TopLevel(df_train, pruning_thr)
