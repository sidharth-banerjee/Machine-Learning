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
def return classArray(examples):
    classes = np.array(examples[:,-1])
    min_class = np.amin(classes)
    return classes, min_class

def returnData(training_file):
    df = pd.read_csv(training_file, delimiter= '\s+', header=None)
    return np.array(df)

def sameClass(examples):
    value = examples[0][-1]
    for i in range (1, len(examples), 1):
        if (examples[i][-1] != value)
            return False
    return True

def class_distribution(examples):
    class_count = [0] * len(examples)
    classes, min_class = classArray(examples)
    for i in range (0, len(examples), 1):
        class_count[classes[i] - min_class] += 1
    class_probability = np.array(class_count)/1000
    return class_probability

def choose_attribute(examples, attributes):
    pass

def DTL_TopLevel(examples, pruning_thr):
    attributes = np.arange(0, len(examples[0])-1, 1)
    classes, min_class = classArray(examples)e
    default = class_distribution(examples)
    return (examples, attributes, default, pruning_thr)

def information_gain(examples, A, threshold):
    pass

def choose_attr_rand(examples, attributes):
    max_gain = best_threshold = -1
    A = random.randint(attributes[0], attributes[-1]+1)
    attribute_values = np.array(examples[:,A])
    L = np.amin(attribute_values)
    M = np.amax(attribute_values)

    for k in range (1, 51, 1):
        threshold = L + K*(M-L)/51
        gain = information_gain(examples, A, threshold)
        if gain > max_gain:
            max_gain = max_gain
            best_threshold = threshold
    return (A, best_threshold)

def choose_attr_Opt(examples, attributes):
    max_gain = best_attribute = best_threshold = -1

    for A in range (0, len(attributes), 1):
        attribute_values = np.array(examples[:,A])
        L = np.amin(attribute_values)
        M = np.amax(attribute_values)

        for k in range (1, 51, 1):
            threshold = L + K*(M-L)/51
            gain = information_gain (examples, A, threshold)
            if gain > max_gain:
                max_gain = max_gain
                best_attribute = A
                best_threshold = threshold
    return (best_attribute, best_threshold)

def DTL(examples, attributes, default, pruning_thr):
    if np.size(examples) < pruning_thr:
        return default

    elif sameClass(examples):
        return examples[0][-1]

    else:
        best_attribute, best_threshold = choose_attribute(examples, attributes)

        tree = Node(best_attribute, best_threshold)
        examples_left = []
        examples_right= []

        #
        for i in range (0, len(examples), 1):
            left = []
            right = []
            for j in range (0, len(examples[i])-1, 1):
                if (examples[i][j] < best_threshold):
                    left.append(examples[i][j])
                else:
                    right.append(examples[i][j])
            left.append(examples[i][-1])
            left.append(examples[i][-1])
            examples_left.append(left)
            examples_right.append(right)
        #

        dist = class_distribution(examples)
        tree.left_child = DTL(examples_left, attributes, dist, pruning_thr)
        tree.right_child = DTL(examples_right, attributes, dist, pruning_thr)

        return tree

# # Main

examples = returnData(training_file)
arr, att, default, pru = DTL_TopLevel(df_train, pruning_thr)
