#!/usr/bin/env python
# coding: utf-8

# In[105]:


'''
Name: Sidharth Banerjee
ID  : 1001622703
'''


# In[106]:


import sys
import random
import numpy as np
import pandas as pd


# In[107]:


env_file = 'environment2.txt'
nt_reward = -0.04
gamma = 1
moves = 1
Ne = 20


# In[108]:


class State:
    def __init__(self, utility, accessible, reward, terminal, ind_i, ind_j):
        self.utility = utility
        self.accessible = accessible
        self.reward = reward
        self.terminal = terminal
        self.ind_i = ind_i
        self.ind_j = ind_j


# In[109]:


def returnTable(file):
    df = pd.read_csv(file, header=None)
    Table = []

    temp0 = []
    for j in range (0, len(df.columns)+2, 1):
        temp0.append(State(0, False, 0, False, -1, -1))
    Table.append(temp0)

    for i in range (0, len(df), 1):
        temp1 = []
        temp1.append(State(0, False, 0, 0, -1, -1))
        for j in range (0, len(df.columns), 1):
            if df.iloc[i][j] == 'X':
                temp1.append(State(0, False, 0, False, i, j))
            elif df.iloc[i][j] == '.':
                temp1.append(State(0, True, nt_reward, False, i, j))
            else:
                temp1.append(State(0, True, float(df.iloc[i][j]), True, i, j))

        temp1.append(State(0, False, 0, False, -1, -1))
        Table.append(temp1)

    Table.append(temp0)
    return Table


# In[110]:


def eta(value):
    return 1/value


# In[111]:


def f(u, n):
    if n < Ne:
        return 1
    else:
        return u


# In[112]:


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


# In[113]:


def ExecuteAction(table, s_, a):
    i = s_.ind_i
    j = s_.ind_j

    def exe_up():
        if table[i-1][j].accessible:
            s_ = table[i-1][j]
        else:
            s_ = table[i][j]
        return s_

    def exe_down():
        if table[i+1][j].accessible:
            s_ = table[i+1][j]
        else:
            s_ = table[i][j]
        return s_

    def exe_left():
        if table[i][j-1].accessible:
            s_ = table[i][j-1]
        else:
            s_ = table[i][j]
        return s_

    def exe_right():
        if table[i][j+1].accessible:
            s_ = table[i][j+1]
        else:
            s_ = table[i][j]
        return s_

    if a == 0:
        rand = weighted_choice()
        if rand == 0:
            s_ = exe_up()
        elif rand == 1:
            s_ = exe_left()
        elif rand == 2:
            s_ = exe_right()

    elif a == 1:
        rand = weighted_choice()
        if rand == 0:
            s_ = exe_down()
        elif rand == 1:
            s_ = exe_left()
        elif rand == 2:
            s_ = exe_right()

    elif a == 2:
        rand = weighted_choice()
        if rand == 0:
            s_ = exe_left()
        elif rand == 1:
            s_ = exe_up()
        elif rand == 2:
            s_ = exe_down()

    elif a == 1:
        rand = weighted_choice()
        if rand == 0:
            s_ = exe_right()
        elif rand == 1:
            s_ = exe_up()
        elif rand == 2:
            s_ = exe_down()

    return s_


# In[114]:


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

        N[s_] = {}
        N[s_][0] = 0
        N[s_][1] = 0
        N[s_][2] = 0
        N[s_][3] = 0

        if s_ in Q:
            action_values = np.array(list(Q[s_].values()))
            Q[s][a] = (1-c)*Q[s][a] + c*(r + gamma*np.amax(action_values))

        else:
            Q[s_] = {}
            Q[s_][0] = 0
            Q[s_][1] = 0
            Q[s_][2] = 0
            Q[s_][3] = 0
            Q[s][a] = (1-c)*Q[s][a] + c*(r)

    elif s is None:
        Q[s] = {}
        Q[s][0] = 0
        Q[s][1] = 0
        Q[s][2] = 0
        Q[s][3] = 0
        Q[s_] = {}
        Q[s_][0] = 0
        Q[s_][1] = 0
        Q[s_][2] = 0
        Q[s_][3] = 0
        N[s_] = {}
        N[s_][0] = 0
        N[s_][1] = 0
        N[s_][2] = 0
        N[s_][3] = 0


    return Q, N


# In[115]:


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
        i_random = None
        j_random = None
        while not(success):
            i_random = random.randint(0, len(table)-1)
            j_random = random.randint(0, len(table[0])-1)
            s_ = table[i_random][j_random]
            if table[i_random][j_random].accessible and not(table[i_random][j_random].terminal):
                success = True

        s_ = table[i_random][j_random]

        while(True):
            r_ = s_.reward

            Q, N = Q_Learning_Update(table, s, r, a, s_, r_, Q, N)

            if s_.terminal:
                break

            u_values = np.array(list(Q[s].values()))

            if s is not None:
                table[s.ind_i][s.ind_j].utility = np.amax(u_values)

            Q_actions = np.array(list(Q[s_].values()))
            N_actions = np.array(list(N[s_].values()))

            f_values = []

            for j in range (0, 4, 1):
                f_values.append(f(Q_actions[i], N_actions[i]))

            a = np.argmax(np.array(f_values))

            s = s_
            r = r_

            s_ = ExecuteAction(table, s_, a)

    return table


# In[ ]:


def printUtilties(U):
    for i in range (1, len(U)-1, 1):
        print('{:6.3f}'.format(U[i][1].utility), end = "")
        for j in range (2, len(U[i])-1, 1):
            print(',{:6.3f}'.format(U[i][j].utility), end = "")
        print()


# In[ ]:


#U = AgentModel_Q_Learning()


# In[ ]:


#printUtilties(U)


# In[ ]:


U_ = returnTable(env_file)
printUtilties(U_)


# In[ ]:
