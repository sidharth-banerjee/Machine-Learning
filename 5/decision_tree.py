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

def returnData(training_file):
    df = pd.read_csv(training_file, delimiter= '\s+', header=None)
    attributes = len(df.columns) - 1
    col_header = np.arange(1, attributes+1, 1)
    col_header = col_header.tolist()
    col_header.append('Class')
    df.columns = col_header
    return df

def sameClass(examples):
    class = examples[0][-1]
    for i in range (1, len(examples), 1):
        if (examples[i][-1] != class)
            return False
    return True

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

def DTL(examples, attributes, default, pruning_thr):
    if np.size(examples) == 0:
        return default

    elif(sameClass(examples)):
        return examples[0][-1]

    best_attribute, best_threshold = choose_attribute(examples, attributes)

    tree = Node(best_attribute, best_threshold)
    examples_left = []
    examples_right= []

    for i in range (0, len(examples), i++):
        if (examples[i][best_attribute] < best_threshold)
            examples_left.append(examples[i][best_attribute])
        else:
            examples_right.append(examples[i][best_attribute])


    tree.left_child = DTL(examples_left, attributes, )

df_train = returnData(training_file)

arr, att, default, pru = DTL_TopLevel(df_train, pruning_thr)
