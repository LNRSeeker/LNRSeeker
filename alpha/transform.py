"""
transform.py
transform the input given the training data
"""
import numpy as np


class Transform:
    def __init__(self, X_train):
        self.X_maxs = np.max(X_train, axis=0)
        self.X_mins = np.min(X_train, axis=0)
        self.X_range = self.X_maxs - self.X_mins
        self.X_range[self.X_range == 0] = 1e-6

    def transform(self, data):
        return (data - self.X_mins) / self.X_range
