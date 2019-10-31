'''
Name: Sidharth Banerjee
ID  : 1001622703
'''

import numpy as np
import pandas as pd
import math
import time
import random


# In[850]:

training_file = str(sys.argv[1])
test_file = str(sys.argv[2])
option = str(sys.argv[3])
pruning_thr= int(sys.argv[4])


# In[851]:


def returnData(training_file):
    df = pd.read_csv(training_file, delimiter= '\s+', header=None)
    return np.array(df)

examples = returnData(training_file)
classes = np.array(examples[:,-1])
low_class = np.amin(classes)
high_class = np.amax(classes)
classSize = int(high_class-low_class+1)


# In[852]:


class Node:
    def __init__(self, attribute, threshold):
        self.attribute   = attribute
        self.threshold   = threshold
        self.gain = None
        self.distribution = None
        self.left_child  = None
        self.right_child = None


# In[853]:


def sameClass(examples):
    for i in range (1, len(examples), 1):
        if examples[i][-1] != examples[i-1][-1]:
            return False
    return True


# In[854]:


def class_distribution(examples):
    class_count = np.zeros(classSize)
    for i in range (0, len(examples), 1):
        class_count[int(examples[i][-1] - low_class)] += 1
    class_probability = np.array(class_count)/len(examples)
    return class_probability


# In[855]:


def information_gain(examples, A, threshold):
    values = np.array(examples[:,A])
    class_dist = class_distribution(examples)

    H_E = 0
    # find entropy at root
    for i in range (0, len(class_dist), 1):
        if class_dist[i] == 0:
            H_E -= 0
        else:
            H_E -= class_dist[i]*math.log2(class_dist[i])

    left = []
    right = []

    for i in range (0, len(values), 1):
        if values[i] < threshold:
            left.append(examples[i])
        else:
            right.append(examples[i])

    # left entropy
    H_left = 0
    if len(left) == 0:
        H_left = 0
    else:
        left_distribution = class_distribution(np.array(left))
        for i in range (0, len(left_distribution), 1):
            if left_distribution[i] == 0:
                H_left -= 0
            else:
                H_left -= left_distribution[i]*math.log2(left_distribution[i])

    # right entropy
    H_right = 0
    if len(right) == 0:
        H_right = 0
    else:
        right_distribution = class_distribution(np.array(right))
        for i in range (0, len(right_distribution), 1):
            if right_distribution[i] == 0:
                H_right -= 0
            else:
                H_right -= right_distribution[i]*math.log2(right_distribution[i])

    # weigh the right and left entropies
    H_left *= len(left)/len(values)
    H_right *= len(right)/len(values)

    info_gain = H_E - H_left - H_right

    return info_gain


# In[856]:


def Optimized(examples, attributes):
    max_gain = best_attribute = best_threshold = -1

    for A in range (0, len(attributes), 1):
        attribute_values = np.array(examples[:,A])
        L = np.amin(attribute_values)
        M = np.amax(attribute_values)

        for K in range (1, 51, 1):
            threshold = L + K*(M-L)/51
            gain = information_gain (examples, A, threshold)
            if gain > max_gain:
                max_gain = gain
                best_attribute = A
                best_threshold = threshold
    return (best_attribute, best_threshold, max_gain)


# In[857]:


def Randomized(examples, attributes):
    max_gain = best_threshold = -1
    A = random.randint(attributes[0], attributes[-1]+1)
    attribute_values = np.array(examples[:,A])
    L = np.amin(attribute_values)
    M = np.amax(attribute_values)

    for K in range (1, 51, 1):
        threshold = L + K*(M-L)/51
        gain = information_gain(examples, A, threshold)
        if gain > max_gain:
            max_gain = gain
            best_threshold = threshold
    return (A, best_threshold, max_gain)


# In[858]:


def choose_attribute(examples, attributes, option):
    if option == 'optimized':
        return Optimized(examples, attributes)
    elif option == 'randomized':
        return (Randomized(examples, attributes))
    elif option == 'forest3' or option == 'forest15':
        return (Randomized(examples, attributes))


# In[859]:


def DTL(examples, attributes, default, pruning_thr, option):
    if len(examples) < pruning_thr:
        x = Node(-1, -1)
        x.distribution = default
        x.gain = 0
        return x

    elif sameClass(examples):
        x = Node(-1, -1)
        cl = examples[0][-1]
        dist = np.zeros(classSize)
        dist[int(cl-low_class)] = 1
        x.distribution = dist
        x.gain = 0
        return x

    else:
        best_attribute, best_threshold, max_gain = choose_attribute(examples, attributes, option)

        tree = Node(best_attribute, best_threshold)
        tree.gain = max_gain
        tree.distribution = class_distribution(examples)

        examples_left = []
        examples_right= []

        for i in range (0, len(examples), 1):
            if examples[i][best_attribute] < best_threshold:
                examples_left.append(examples[i])
            else:
                examples_right.append(examples[i])

        examples_left = np.array(examples_left)
        examples_right = np.array(examples_right)

        tree.left_child = DTL(examples_left, attributes, tree.distribution, pruning_thr, option)
        tree.right_child = DTL(examples_right, attributes, tree.distribution, pruning_thr, option)

        return tree


# In[860]:


def DTL_TopLevel(examples, pruning_thr, option):
    attributes = np.arange(0, len(examples[0])-1, 1)
    default = class_distribution(examples)
    return DTL(examples, attributes, default, 50, option)


# In[861]:


trees = []
tree = DTL_TopLevel(examples, pruning_thr, option)
trees.append(tree)

if option == 'forest3':
    for i in range(0, 2, 1):
        trees.append(DTL_TopLevel(examples, pruning_thr, option))
elif option == 'forest15':
    for i in range(0, 14, 1):
        trees.append(DTL_TopLevel(examples, pruning_thr, option))


# In[862]:


index = 1
num = 1
def printLevelOrder(root):
    h = height(root)
    for i in range(1, h+1):
        printGivenLevel(root, i)

def printGivenLevel(root , level):
    if root is None:
        return
    if level == 1:
        global index
        global num
        print("tree={:2d}, node={:3d}, feature={:2d}, thr={:6.2f}, gain={:f}"
              .format(num, index, root.attribute, root.threshold, root.gain))
        index += 1

    elif level > 1 :
        printGivenLevel(root.left_child, level-1)
        printGivenLevel(root.right_child, level-1)

def height(node):
    if node is None:
        return 0
    else :
        # Compute the height of each subtree
        lheight = height(node.left_child)
        rheight = height(node.right_child)

        #Use the larger one
        if lheight > rheight :
            return lheight+1
        else:
            return rheight+1


# In[863]:


for i in range (0, len(trees), 1):
    global index
    index = 1
    global num
    print('Print tree {:2d}\n'.format(i+1))
    printLevelOrder(trees[i])
    print('\n********\n')
    num += 1


# In[864]:


def answer(x, tree):
    if tree.left_child is None and tree.right_child is None:
        return tree.distribution
    elif tree.right_child is None:
        return answer(x, tree.left_child)
    elif tree.left_child is None:
        return answer(x, tree.right_child)
    else:
        if x[tree.attribute] < tree.threshold:
            return answer(x, tree.left_child)
        else:
            return answer(x, tree.right_child)


# In[865]:


test = returnData(test_file)


# In[866]:


class_acc = []
print()
for i in range (0, len(test), 1):
    d = -1
    if (len(trees) == 1):
        a = answer(test[i], trees[0])
        d = np.argmax(a)+low_class

    else:
        avg = []
        for j in range(0, len(trees), 1):
            a = answer(test[i], trees[j])
            avg.append(np.argmax(a))
        avg = np.array(avg)
        d = np.mean(avg) + low_class

    if int(d) == int(test[i][-1]):
        class_acc.append(1)
    else:
        class_acc.append(0)

    print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}"
              .format(i, int(d), int(test[i][-1]), class_acc[i]))


# In[867]:


print()
class_acc = np.array(class_acc)
print('classification accuracy={:6.4f}'.format(np.mean(class_acc)))


# In[ ]:
