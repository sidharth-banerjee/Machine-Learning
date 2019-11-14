'''
Name: Sidharth Banerjee
ID  : 1001622703
'''

import sys
import math
import random
import numpy as np
import pandas as pd

# Command Line Arguments
data_file = str(sys.argv[1])
k = int(sys.argv[2])
iterations = int(sys.argv[3])

# ## Functions

def returnData(fileName):
    df = pd.read_csv(data_file, header=None)
    df = df[df.columns[:-1]]
    df = df.astype(float)
    return df

def init(df):
    weights = np.zeros(k)
    mean = [[0 for x in range(len(df.columns))] for y in range (k)]
    covariance = []
    cov = [[0 for x in range(len(df.columns))] for y in range (len(df.columns))]
    for i in range (0, k, 1):
        covariance.append(cov)
    return(np.array(weights), np.array(mean), np.array(covariance))

def setProbability(df):
    p = []
    for j in range (0, len(df), 1):
        p_x = []
        c = random.randint(0, k-1)
        for i in range (0, k, 1):
            if i == c:
                p_x.append(1)
            else:
                p_x.append(0)
        p.append(p_x)
    return(np.array(p))

def multiGaussian(df, mean, covariance):
    dim = len(df)
    constant = (2*np.pi)**dim
    constant *= np.linalg.det(covariance)
    constant = 1/(math.sqrt(constant))
    N = np.linalg.multi_dot([(df.iloc[0] - mean).T, np.linalg.inv(covariance), df.iloc[0] - mean])
    N = constant*np.exp(-0.5*N)
    return N

def eStage(df, weights, mean, covariance):
    N = []
    for i in range (0, k, 1):
        gaus_k = []
        for j in range (0, len(df), 1):
            gaus_k.append(multiGaussian(df.iloc[j], mean[i], covariance[i]))
        N.append(gaus_k)
    N = np.array(N).T
    new_prob = []

    for j in range (0, len(df), 1):
        j_sum = 0
        j_prob = []
        for i in range (0, k, 1):
            j_sum += N[j][i]*weights[i]

        for i in range (0, k, 1):
            p[j][i] = (N[j][i]*weights[i])/j_sum
            j_prob.append((N[j][i]*weights[i])/j_sum)
        new_prob.append(j_prob)
    return np.array(new_prob)

def mStage(df, p, weights, mean, covariance):

    # update weights
    for i in range (0, k, 1):
        value = 0
        for j in range (0, len(df), 1):
            value += p[j][i]
        weights[i] = value

    sum_weights = np.sum(weights)

    for i in range (0, k, 1):
        weights[i] = weights[i]/sum_weights

    # update mean
    for i in range (0, k, 1):
        for d in range (0, len(df.columns), 1):
            num = 0
            den = 0
            for j in range (0, len(df), 1):
                num += p[j][i]*df.iloc[j][d]
                den += p[j][i]
            mean[i][d] = num/den

    # update covariance matrix
    for i in range (0, k, 1):
        for r in range (0, len(df.columns), 1):
            for c in range (0, len(df.columns), 1):
                num = 0
                den = 0
                for j in range (0, len(df), 1):
                    num += p[j][i]*(df.iloc[j][r] - mean[i][r])*(df.iloc[j][c] - mean[i][c])
                    den += p[j][i]
                if r==c and num/den < 0.0001:
                    covariance[i][r][c] = 0.0001
                else:
                    covariance[i][r][c] = num/den
    return weights, mean, covariance

def inter_output(weights, mean):
    for i in range (0, k, 1):

        print('weight {:d} = {:.4f}, '.format(i+1, weights[i]), end = '')
        print('mean {:d} = ('.format(i+1), end = '')
        print('{:.4f}'.format(mean[i][0]), end = '')
        for j in range (1, len(mean[i]), 1):
            print(', {:.4f}'.format(mean[i][j]), end = '')
        print(')')

def final_output(weights, mean, covariance):
    for i in range (0, k, 1):
        print('weight {:d} = {:.4f}, '.format(i+1, weights[i]), end = '')
        print('mean {:d} = ('.format(i+1), end = '')
        print('{:.4f}'.format(mean[i][0]), end = '')

        for j in range (1, len(mean[i]), 1):
            print(', {:.4f}'.format(mean[i][j]), end = '')
        print(')')

        for j in range (0, len(covariance[i]), 1):
            print('Sigma {:d} row {:d} = '.format(i+1, j+1), end = '')
            print('{:.4f}'.format(covariance[i][j][0]), end = '')
            for m in range (1, len(covariance[i][j]), 1):
                print(', {:.4f}'.format(covariance[i][j][m]), end = '')
            print()


# ## Main

df = returnData(data_file)

p = setProbability(df)
weights, mean, covariance = init(df) # initializes shape of variables

for i in range (0, iterations, 1):
    weights, mean, covariance = mStage(df, p, weights, mean, covariance)
    p = eStage(df, weights, mean, covariance)

    if i < iterations-1:
        print('After iteration {:d}:'.format(i+1))
        inter_output(weights, mean)
    else:
        print('After final iteration:')
        final_output(weights, mean, covariance)
    print()
