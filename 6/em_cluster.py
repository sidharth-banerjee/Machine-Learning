#!/usr/bin/env python
# coding: utf-8

# In[25]:


'''
Name: Sidharth Banerjee
ID  : 1001622703
'''


# In[26]:


import sys
import math
import random
import numpy as np
import pandas as pd


# In[27]:


# Command Line Arguments
data_file = 'point_set1.txt'
k = 3
iterations = 5


# ## Functions

# In[28]:


def returnData(fileName):
    df = pd.read_csv(data_file, header=None)
    df = df[df.columns[:-1]]
    df = df.astype(float)
    return df


# In[29]:


def init(df):
    weights = np.zeros(k)
    mean = [[0 for x in range(len(df.columns))] for y in range (k)]
    covariance = []
    cov = [[0 for x in range(len(df.columns))] for y in range (len(df.columns))]
    for i in range (0, k, 1):
        covariance.append(cov)
    return(np.array(weights), np.array(mean), np.array(covariance))


# In[30]:


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


# In[31]:


def multiGaussian(df, mean, covariance):
    dim = len(df)
    constant = (2*np.pi)**dim
    constant *= np.linalg.det(covariance)
    constant = 1/(math.sqrt(constant))
    N = np.linalg.multi_dot([(df.iloc[0] - mean).T, np.linalg.inv(covariance), df.iloc[0] - mean])
    N = constant*np.exp(-0.5*N)
    return N


# In[32]:


def eStage(df, p, weights, mean, covariance):
    N = []
    for i in range (0, k, 1):
        gaus_k = []
        for j in range (0, len(df), 1):
            gaus_k.append(multiGaussian(df.iloc[j], mean[i], covariance[i]))
        N.append(gaus_k)
    N = np.array(N).T

    for j in range (0, len(df), 1):
        j_sum = 0
        for i in range (0, k, 1):
            j_sum += N[j][i]*weights[i]
            
        for i in range (0, k, 1):
            p[j][i] = (N[j][i]*weights[i])/j_sum
            print((N[j][i]*weights[i])/j_sum, p[j][i])
    
    return p


# In[33]:


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


# ## Main

# In[34]:


df = returnData(data_file)


# In[35]:


p = setProbability(df)
weights, mean, covariance = init(df) # initializes shape of variables


# In[36]:


for i in range (0, 1, 1):
    weights, mean, covariance = mStage(df, p, weights, mean, covariance)
    p = eStage(df, p, weights, mean, covariance)
    #print(p)


# In[ ]:





# In[ ]:




