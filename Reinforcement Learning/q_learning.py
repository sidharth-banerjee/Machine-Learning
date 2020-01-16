'''
Name: Sidharth Banerjee
ID  : 1001622703
'''

import sys
import random
import numpy as np
import pandas as pd

env_file = 'environment2.txt'
nt_reward = -0.04
gamma = 1
moves = 1000
Ne = 20

class State:
    def __init__(self, utility, accessible, reward, terminal):
        self.utility = utility
        self.accessible = accessible
        self.reward = reward
        self.terminal = terminal

def returnTable(file):
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

def eta(value):
    return 1/value

def f(u, n):
    if n < Ne:
        return 1
    else:
        return u

# weighted random number generator, with probabilities as weights
def weighted_choice():
       weights = np.array([0.8, 0.1, 0.1])
       totals = []
       running_total = 0

       for w in weights:
           running_total += w
           totals.append(running_total)

       rnd = random.random() * running_total
       for i, total in enumerate(totals):
           if rnd < total:
               return i

def ExecuteAction(table, s_, a, i, j):

    def exe_up(i, j):
        if table[i-1][j].accessible:
            s_ = table[i-1][j]
            i = i-1
        else:
            s_ = table[i][j]
        return s_, i, j

    def exe_down(i, j):
        if table[i+1][j].accessible:
            s_ = table[i+1][j]
            i = i +1
        else:
            s_ = table[i][j]
        return s_, i, j

    def exe_left(i, j):
        if table[i][j-1].accessible:
            s_ = table[i][j-1]
            j = j -1
        else:
            s_ = table[i][j]
        return s_, i, j

    def exe_right(i, j):
        if table[i][j+1].accessible:
            s_ = table[i][j+1]
            j = j+1
        else:
            s_ = table[i][j]
        return s_, i, j

    if a == 0:
        rand = weighted_choice()
        if rand == 0:
            s_, i, j = exe_up(i, j)
        elif rand == 1:
            s_, i, j = exe_left(i, j)
        elif rand == 2:
            s_, i, j = exe_right(i, j)

    elif a == 1:
        rand = weighted_choice()
        if rand == 0:
            s_, i, j = exe_down(i, j)
        elif rand == 1:
            s_, i, j = exe_left(i, j)
        elif rand == 2:
            s_, i, j = exe_right(i, j)

    elif a == 2:
        rand = weighted_choice()
        if rand == 0:
            s_, i, j_ = exe_left(i, j)
        elif rand == 1:
            s_, i, j = exe_up(i, j)
        elif rand == 2:
            s_, i, j = exe_down(i, j)

    elif a == 1:
        rand = weighted_choice()
        if rand == 0:
            s_, i, j = exe_right(i, j)
        elif rand == 1:
            s_, i, j = exe_up(i, j)
        elif rand == 2:
            s_, i, j = exe_down(i, j)

    return s_, i, j

def Q_Learning_Update(table, s, r, a, s_, r_, Q, N):
    if s_.terminal:
        Q[s_] = {}
        Q[s_][None] = r_

    if s is not None:
        if s in N and a in N[s]:
            N[s][a] += 1
        else:
            N[s] = {}
            N[s][a] = 1
            Q[s] = {}

        c = eta(N[s][a])

        if s_ not in Q:
            Q[s_] = {}
            Q[s_][a] = 0
            
        action_values = np.array(list(Q[s_].values()))

        Q[s][a] = (1-c)*Q[s][a] + c*(r + gamma*np.amax(action_values))

    return Q, N

def AgentModel_Q_Learning():
    table = returnTable(env_file)
    num_of_states = len(table)*len(table[0])
    Q = {}
    N = {}

    for i in range (0, moves, 1):
        s = None
        r = None
        a = None

        success = False
        s_ = None
        i_prev = None
        j_prev = None
        i_next = None
        j_next = None
        while not(success):
            i_next = random.randint(0, len(table)-1)
            j_next = random.randint(0, len(table[0])-1)
            s_ = table[i_next][j_next]
            if table[i_next][j_next].accessible and not(table[i_next][j_next].terminal):
                success = True

        s_ = table[i_next][j_next]

        while(True):
            r_ = s_.reward

            if s is None:
                a = 0
                s = s_
                s_, i_next, j_next = ExecuteAction(table, s_, a, i_next, j_next)

            else:
                Q, N = Q_Learning_Update(table, s, r, a, s_, r_, Q, N)

                if s_.terminal:
                    break

                u_values = np.array(list(Q[s].values()))

                if s is not None:
                    table[i_prev][j_prev].utility = np.amax(u_values)

                Q_actions = np.array(list(Q[s_].values()))
                N_actions = np.array(list(N[s_].values()))

                f_values = []

                for j in range (0, 4, 1):
                    f_values.append(f(Q_actions[j], N_actions[j]))

                a = np.argmax(np.array(f_values))

                s = s_
                i_prev = i_next
                j_prev = j_next

                s_, i_next, j_next = ExecuteAction(table, s_, a, i_next, j_next)

    return table

def printUtilties(U):
    for i in range (1, len(U)-1, 1):
        print('{:6.3f}'.format(U[i][1].accessible), end = "")
        for j in range (2, len(U[i])-1, 1):
            print(',{:6.3f}'.format(U[i][j].accessible), end = "")
        print()

U = AgentModel_Q_Learning()

printUtilties(U)
