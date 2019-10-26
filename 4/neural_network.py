'''
Name: Sidharth Banerjee
ID  : 1001622703
'''

import sys
import math
import random
import numpy as np
import pandas as pd
from scipy.sparse import bsr_matrix

# command line arguments
training_file = str(sys.argv[1])
test_file = str(sys.argv[2])
layers = int(sys.argv[3])
units_per_layer = int(sys.argv[4])
rounds = int(sys.argv[5])

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

# normalize data-set
def normalize(df):
    classes  = np.array(df['Class'])
    maximum  = df.iloc[ :, :-1].max().max()
    df_norm  = df.iloc[ :, :-1]/maximum
    df_norm['Class'] = classes
    return df_norm

# Returns a random real number
def randomWeight():
    return  random.uniform(-0.05, 0.05)

# Return a Sparse matrix of shape [UxU]
def returnAdjacency(U):
    adj = bsr_matrix((U, U), dtype=np.float).toarray()
    return adj

# Function to return index of first unit in every layer
def returnIndices(df, U, class_size, layers, units_per_layer):
    x = np.array(df.iloc[0][:-1])
    indices = []

    # 1st unit index in layer 1
    indices.append(1)

    # if no hidden layers
    if (layers == 2):
        indices.append(dimensions) #unit no. for layer 2
        return indices

    # hidden layers exist
    # for each hidden layer, determine unit index
    for l in range (0, layers-2, 1):
        index = dimensions + l*units_per_layer
        indices.append(index)

    # start index of output layer
    indices.append(U-class_size)
    return indices

def returnWeights(df, t, U, class_size, rounds, indices):
    weights = []

    for n in range (0, len(df), 1):
        z = np.zeros(U)
        r = U - class_size
        x = np.array(df_train.iloc[n][:-1])
        target = t[n]
        w = returnAdjacency(U)

        for l in range (1, layers, 1):
            if (l < layers - 1):
                for j in range (indices[l], indices[l+1], 1):
                    w[0][j] = randomWeight()
                    for i in range (indices[l-1], indices[l], 1):
                        w[i][j] = randomWeight()
            # last layer
            else:
                for j in range (indices[l], U, 1):
                    w[0][j] = randomWeight()
                    for i in range (indices[l-1], indices[l], 1):
                        w[i][j] = randomWeight()

        # set output of units in layer 1
        for j in range (0, dimensions, 1):
            z[j] = (x[j])
        # update output of units in subsequent layers
        a = np.zeros(U)

        for l in range (1, layers, 1):
            if l < layers - 1:
                for j in range (indices[l], indices[l+1], 1):
                    sigma = 0
                    sigma += w[0][j]*z[0]
                    for i in range (indices[l-1], indices[l], 1):
                        sigma+= w[i][j]*z[i]
                    a[j] = sigma
                    z[j] = ( 1/(1+np.exp(-a[j])))

            # last layer
            else:
                for j in range (indices[l], U, 1):
                    sigma = 0
                    sigma += w[0][j]*z[0]
                    for i in range (indices[l-1], indices[l], 1):
                        sigma += w[i][j]*z[i]
                    a[j] = (sigma)
                    z[j] = (1/(1+np.exp(-a[j])))

        delta = np.zeros(U)

        # for each output unit
        for j in range (r, U, 1):
            delta[j] = (z[j] - target[j-r]) * z[j] * (1-z[j])

        # update previous layers
        for l in range (layers - 2, 0, -1):
            for j in range (indices[l], indices[l+1], -1):
                sigma = 0
                sigma += delta[j]*w[0][j]
                for i in range (indices[l], indices[l+1], 1):
                    sigma+= delta[i]*w[i][j]
                delta[j] = sigma*z[j]*(1-z[j])

        for Rounds in range (0, rounds, 1):
            eta =  math.pow(0.98, Rounds-1)
            # update weights
            for l in range (1, layers, 1):
                if (l < layers - 1):
                    for j in range (indices[l], indices[l+1], 1):
                        w[0][j] = w[0][j] - eta*delta[j]*z[0]
                        for i in range (indices[l-1], indices[l], 1):
                            w[i][j] = w[i][j] - eta*delta[j]*z[i]

                # last layer
                else:
                    for j in range (indices[l], U, 1):
                        w[0][j] = w[0][j] - eta*delta[j]*z[0]
                        for i in range (indices[l-1], indices[l], 1):
                            w[i][j] = w[i][j] - eta*delta[j]*z[i]

        weights.append(w)
    return weights

def returnAccuracy(z, target, min_class):
    values = np.argwhere(z == np.amax(z))
    values = values.flatten().tolist()
    answer = -1
    if (len(values) > 1):
        answer = values[int(random.random()%len(values))]
    else:
        answer = values[0]

    predicted_class = answer + min_class

    class_difference = abs(predicted_class - target)

    accuracy = float(-1)
    if class_difference == 0 and len(values) == 1:
        accuracy = 1
        return predicted_class, accuracy
    elif class_difference > 0 and len(values) == 1:
        accuracy = 0
        return predicted_class, accuracy
    elif len(values) > 1:
        duplicates = 0
        for i in range (0, len(values), 1):
            class_difference = abs(values[i]+1 - target)
            if (class_difference == 0):
                accuracy = 1/len(values)
                return predicted_class, accuracy
        accuracy = 0
        return predicted_class, accuracy

# # Main

#load training data
df_train = normalize(returnData(training_file))
#add bias input
bias_input = np.ones(len(df_train))
df_train.insert(0, 0, bias_input)
dimensions = len(df_train.columns)-1
max_class = df_train['Class'].max()
min_class = df_train['Class'].min()
class_size = max_class - min_class + 1

# # Training Stage

# one-versus all vector
t = []
for i in range (0, len(df_train), 1):
    vector = np.zeros(class_size)
    vector[int(df_train.iloc[i]['Class'])-min_class] = 1
    t.append(vector)
t = np.array(t)

# backpropagation
U = dimensions + class_size + (layers-2)*(units_per_layer) #total units

indices = np.array(returnIndices(df_train, U, class_size, layers, units_per_layer))

weights = returnWeights(df_train, t, U, class_size, rounds, indices)

# # Classification Stage

# load test data
df_test = normalize(returnData(test_file))
bias_input = np.ones(len(df_test))
df_test.insert(0, 0, bias_input)

predicted = []
accuracy  = []
for n in range (0, len(df_test), 1):
    U = dimensions + class_size + (layers-2)*(units_per_layer)
    z = np.empty(U, dtype=float)
    a = np.empty(U, dtype=float)
    r = U - class_size
    x = np.array(df_test.iloc[n][:-1])
    target = df_test.iloc[n]['Class']

    for j in range (0, dimensions, 1):
            z[j] = x[j]

    for l in range (1, layers, 1):
            if l < layers - 1:
                for j in range (indices[l], indices[l+1], 1):
                    sigma = 0
                    sigma+=weights[n][0][j]*z[0]
                    for i in range (indices[l-1], indices[l], 1):
                        sigma+= weights[n][i][j]*z[i]
                    a[j] = sigma
                    z[j] = 1/(1+np.exp(-a[j]))

            # last layer
            else:
                for j in range (indices[l], U, 1):
                    sigma = 0
                    sigma+=weights[n][0][j]*z[0]
                    for i in range (indices[l-1], indices[l], 1):
                        sigma += weights[n][i][j]*z[i]
                    a[j] = sigma
                    z[j] = 1/(1+np.exp(-a[j]))
    z = z[r:U]
    CLASS, ACC = returnAccuracy(z, target, min_class)
    predicted.append(CLASS)
    accuracy.append(ACC)

for n in range (0, len(df_test), 1):
    print("{:5d}, {:3d}, {:3d}, {:4.2f}".format(n+1, int(predicted[n]),
                                                         int(df_test.iloc[n]['Class']), accuracy[n]))

print()
accuracy = np.array(accuracy)
print('classification accuracy = {:6.4f}\n'.format(accuracy.mean()))
