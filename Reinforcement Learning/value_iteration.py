'''
Name: Sidharth Banerjee
ID  : 1001622703
'''

import sys
import copy
import numpy as np
import pandas as pd

# command line arguments

env_file = str(sys.argv[1])
nt_reward = float(sys.argv[2])
gamma = float(sys.argv[3])
K = int(sys.argv[4])

# methods

class State:
    def __init__(self, utility, accessible, reward, terminal):
        self.utility = utility
        self.accessible = accessible
        self.reward = reward
        self.terminal = terminal

def returnUtility(file):
    df = pd.read_csv(file, header=None)
    Table = []

    temp0 = []
    for j in range (0, len(df.columns)+2, 1):
        temp0.append(State(0, False, 0, False))
    Table.append(temp0)

    for i in range (0, len(df), 1):
        temp1 = []
        temp1.append(State(0, False, 0, 0))
        for j in range (0, len(df.columns), 1):
            if df.iloc[i][j] == 'X':
                temp1.append(State(0, False, 0, False))
            elif df.iloc[i][j] == '.':
                temp1.append(State(0, True, nt_reward, False))
            else:
                temp1.append(State(0, True, float(df.iloc[i][j]), True))

        temp1.append(State(0, False, 0, False))
        Table.append(temp1)

    Table.append(temp0)
    return Table

def action_up(U, i, j):
    UP = 0.0

    if U[i-1][j].accessible:
        UP += 0.8*U[i-1][j].utility
    if not(U[i-1][j].accessible):
        UP += 0.8*U[i][j].utility

    if U[i][j-1].accessible:
        UP += 0.1*U[i][j-1].utility
    if not(U[i][j-1].accessible):
        UP += 0.1*U[i][j].utility

    if U[i][j+1].accessible:
        UP += 0.1*U[i][j+1].utility
    if not(U[i][j+1].accessible):
        UP += 0.1*U[i][j].utility

    return UP

def action_down(U, i, j):
    DOWN = 0.0

    if U[i+1][j].accessible:
        DOWN += 0.8*U[i+1][j].utility
    if not(U[i+1][j].accessible):
        DOWN += 0.8*U[i][j].utility

    if U[i][j-1].accessible:
        DOWN += 0.1*U[i][j-1].utility
    if not(U[i][j-1].accessible):
        DOWN += 0.1*U[i][j].utility

    if U[i][j+1].accessible:
        DOWN += 0.1*U[i][j+1].utility
    if not(U[i][j+1].accessible):
        DOWN += 0.1*U[i][j].utility

    return DOWN

def action_left(U, i, j):
    LEFT = 0.0

    if U[i][j-1].accessible:
        LEFT += 0.8*U[i][j-1].utility
    if not(U[i][j-1].accessible):
        LEFT += 0.8*U[i][j].utility

    if U[i-1][j].accessible:
        LEFT += 0.1*U[i-1][j].utility
    if not(U[i-1][j].accessible):
        LEFT += 0.1*U[i][j].utility

    if U[i+1][j].accessible:
        LEFT += 0.1*U[i+1][j].utility
    if not(U[i+1][j].accessible):
        LEFT += 0.1*U[i][j].utility

    return LEFT

def action_right(U, i, j):
    RIGHT = 0.0

    if U[i][j+1].accessible:
        RIGHT += 0.8*U[i][j+1].utility
    if not(U[i][j+1].accessible):
        RIGHT += 0.8*U[i][j].utility

    if U[i-1][j].accessible:
        RIGHT += 0.1*U[i-1][j].utility
    if not(U[i-1][j].accessible):
        RIGHT += 0.1*U[i][j].utility

    if U[i+1][j].accessible:
        RIGHT += 0.1*U[i+1][j].utility
    if not(U[i+1][j].accessible):
        RIGHT += 0.1*U[i][j].utility

    return RIGHT

def findMax(U, i, j):
    Utilities = []

    Utilities.append(action_up(U, i, j))
    Utilities.append(action_down(U, i, j))
    Utilities.append(action_left(U, i, j))
    Utilities.append(action_right(U, i, j))

    return np.amax(np.array(Utilities))

def ValueIteration():
    U_ = returnUtility(env_file)

    for i in range (0, K, 1):
        U = copy.deepcopy(U_)

        for i in range (1, len(U_)-1, 1):
            for j in range (1, len(U_[i])-1, 1):
                if U_[i][j].terminal or not(U_[i][j].accessible):
                    U_[i][j].utility = U_[i][j].reward
                else:
                    maxAction = findMax(U, i, j)
                    U_[i][j].utility = U_[i][j].reward + gamma*maxAction
    return U_

def printUtilties(U):
    for i in range (1, len(U)-1, 1):
        print('{:6.3f}'.format(U[i][1].utility), end = "")
        for j in range (2, len(U[i])-1, 1):
            print(',{:6.3f}'.format(U[i][j].utility), end = "")
        print()


# main

U = ValueIteration()
printUtilties(U)
