'''
Name: Sidharth Banerjee
ID  : 1001622703
Date: 09/12/2019
'''

import sys
import math
import random
import pandas as pd
import numpy  as np


# # Data Pre-processing

training_file = str(sys.argv[1])
test_file = str(sys.argv[2])

df = pd.read_csv(training_file, delimiter= '\s+', header=None)
attributes = len(df.columns) - 1
col_header = np.arange(1, attributes+1, 1)
col_header = col_header.tolist()
col_header.append('Class')
values = len(df)
df.columns = col_header
class_H = df['Class'].max()
class_L = df['Class'].min()
class_size = class_H - class_L + 1


# # Training

gaussian_list = []

for CLASS in range (class_L, class_H+1, 1):
    df_temp = df.where(df['Class'] == CLASS)
    for ATTRIBUTE in range (1, attributes+1, 1):
        mean_temp = df_temp[ATTRIBUTE].mean()
        std_temp  = df_temp[ATTRIBUTE].std()
        if (std_temp < 0.01):
            std_temp = 0.01
        gaussian_list.append([CLASS, ATTRIBUTE, mean_temp, std_temp])

df2 = pd.DataFrame(gaussian_list)
col_head = ['Class', 'Attribute', 'Mean', 'Standard Deviation']
df2.columns = col_head
df2['Mean'] = df2['Mean'].map('{:,.2f}'.format)
df2['Standard Deviation'] = df2['Standard Deviation'].map('{:,.2f}'.format)

print('Training Stage:\n')
for i in range (0, len(df2), 1):
    print("{:d}, {:d}, {:.2f}, {:.2f}".format(int(df2.iloc[i]['Class']), int(df2.iloc[i]['Attribute']),
                                                float(df2.iloc[i]['Mean']), float(df2.iloc[i]['Standard Deviation'])))

print()
# # Classification

print('Classification Stage:\n')
# pre-processing test file

df3 = pd.read_csv(test_file, delimiter= '\s+', header=None)
df3.columns = col_header

# p(C)

class_count = [0] * (df['Class'].max())
for i in range (0, len(df)):
    class_count[int(df.iloc[i]['Class'])-1] += 1

class_count = np.array(class_count)
class_probability = class_count/1000

# P(X|C) and P(X)

def return_gaussian(x, u, sigma):
    exponent = math.exp(-(math.pow(x-u,2)/(2*math.pow(sigma,2))))
    return (1 / (math.sqrt(2*math.pi) * sigma)) * exponent

result = []

for OBJECT in range(0, len(df3)):
    class_gaussian = []
    for CLASS in range(class_L, class_H+1, 1):
        temp_gaussian = 1
        for ATTRIBUTE in range(1, attributes+1, 1):
            df_temp2 = df2.where(df2['Class'] == CLASS).dropna()
            df_temp2 = df_temp2.where((df_temp2['Attribute']) == float(ATTRIBUTE)).dropna()
            u = df_temp2['Mean']
            sigma = df_temp2['Standard Deviation']
            temp_gaussian *= return_gaussian(float(df3.iloc[OBJECT][ATTRIBUTE]), float(u), float(sigma))
        class_gaussian.append(temp_gaussian)

    P_x = 0
    for i in range (0, class_size, 1):
        P_x += (class_gaussian[i]*class_probability[i])

    P_c_x =[]
    for i in range (0, class_size, 1):
        P_c_x.append((class_gaussian[i]*class_probability[i])/P_x)

    b_x = np.argwhere(P_c_x == np.amax(P_c_x))
    b_x = b_x.flatten().tolist()

    B_x = -1
    if (len(b_x) > 1):
        B_x = b_x[int(random.random()%len(b_x))]
    else:
        B_x = b_x[0]

    predicted_class = B_x+1
    real_class = int(df3.iloc[OBJECT]['Class'])

    class_difference = abs(predicted_class - real_class)

    accuracy = float(-1)
    if (class_difference == 0 and len(b_x) == 1):
        accuracy = 1
    elif (class_difference > 0 and len(b_x) == 1):
        accuracy = 0
    elif (len(b_x) > 1):
        duplicates = 0
        for i in range (0, len(b_x), 1):
            class_difference = abs(b_x[i]+1 - real_class)
            if (class_difference == 0):
                duplicates += 1

        if (duplicates > 0):
            accuracy = 1/duplicates
        else:
            accuracy = 0

    result.append([OBJECT+1, predicted_class , P_c_x[B_x], real_class, accuracy])
    print("{:5d}, {:3d}, {:.4f}, {:3d}, {:4.2f}".format(OBJECT+1, predicted_class, P_c_x[B_x], real_class, accuracy))

result = pd.DataFrame(result)
result.columns = ['ID', 'Predicted', 'Probability', 'True', 'Accuracy']

print()
print('\nClassification accuracy= {:6.4f}'.format(float(result['Accuracy'].mean())))
