
'''
Name: Sidharth Banerjee
ID  : 1001622703
Date: 09/26/2019
'''

import sys
import numpy as np
import pandas as pd

training_file = str(sys.argv[1])
degree = int(sys.argv[2])
Lambda = int(sys.argv[3])
test_file = str(sys.argv[4])

# load training data
df_train = pd.read_csv(training_file, delimiter= '\s+', header=None)
attributes = len(df_train.columns) - 1
col_header = np.arange(0, attributes, 1)
col_header = col_header.tolist()
col_header.append('Class')
df_train.columns = col_header
df_train = df_train.drop(['Class'], axis=1)

# t array
t = np.array(df_train[df_train.columns[-1]].T)
t = t.reshape((len(t), 1))

# phi array
phi_train = []
phi_train.append([1]*len(df_train[0]))
for i in range (0, len(df_train.columns)-1, 1):
    phi_train.append(np.array(df_train[i]))
    if (degree > 1):
        for j in range (2, degree+1, 1):
            phi_train.append(df_train[i]**j)

phi_train = np.array(phi_train).T

# calculate weights
print('Training Stage')
I = np.identity(len(df_train.columns)*degree-1*(degree-1))
w = (np.linalg.pinv(Lambda*I+(phi_train.T)@(phi_train)))@(phi_train.T)@t
for i in range (0, len(w), 1):
     print("w{:1d} = {:.4f}".format(int(i), float(w[i])))

# load test data
df_test = pd.read_csv(test_file, delimiter= '\s+', header=None)
df_test.columns = col_header
df_test = df_test.drop(['Class'], axis=1)

phi_test = []
phi_test.append([1]*len(df_test[0]))
for i in range (0, len(df_test.columns)-1, 1):
    phi_test.append(np.array(df_test[i]))
    if (degree > 1):
        for j in range (2, degree+1, 1):
            phi_test.append(df_test[i]**j)
phi_test = np.array(phi_test).T

y_predicted = []
y_actual = []
squared_error = []
print('\nTest Stage')
for i in range (0, len(df_test), 1):
    y = np.dot(w.T, phi_test[i])
    y_predicted.append(y)
    y_actual.append(df_test.iloc[i][len(df_test.columns)-1])
    squared_error.append((y_predicted[i] - y_actual[i])**2)
    print("ID={:5d}, output={:14.2f}, target value={:10.2f}, squared error={:.4f}".format(int(i+1),
                                                                                   float(y_predicted[i]),
                                                                                   float(y_actual[i]),
                                                                                   float(squared_error[i])))
