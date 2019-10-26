'''
Class Implementation for classes Node and Tree
'''

class Node(object):
    def _init_(self, data):
        self.data = data
        self.threshold = None
        self.left = None
        self.right = None

    def threshold(self, x):
        L = np.amax(x)
        M = np.amin(x)

    def insert(self, data):
        if data < self.threshold:
            self.left = data
        else:
            self.right = data
