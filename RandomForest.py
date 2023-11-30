from DecisionTree import DecisionTree
import numpy as np
import random

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def subset(self, X, y, n_features=30):
        # Random sample with replacement n_features from X and y
        X_y = np.concatenate((X, y), axis=1)
        X_y_subset = np.random.choice(X_y.shape[0], size=n_features, replace=True)
        X_y_subset = X_y[X_y_subset, :]
        print(X_y_subset)