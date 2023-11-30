from DecisionTree import DecisionTree
import numpy as np
import random

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def subset(self, X, y, size=200):
        # Random sample with replacement n_features from X and y
        X_y = np.concatenate((X, y), axis=1)
        X_y_subset = np.random.choice(X_y.shape[0], size=size, replace=True)
        X_y_subset = X_y[X_y_subset, :]
        return X_y_subset
    
    def fit(self, X, y, size = 100, n_variables = 3):
        for i in range(self.n_estimators):
            X_y_subset = self.subset(X, y, size)

            # choose n_variables from n_features randomly
            selected_columns = np.random.choice(X_y_subset.shape[1] - 1, 3, replace=False)

            # Keep the last column
            selected_columns = np.append(selected_columns, X_y_subset.shape[1] - 1)

            selected_data = X_y_subset[:, selected_columns]

            X_subset = selected_data[:, :-1]
            y_subset = selected_data[:, -1].reshape(-1,1)
            self.trees.append(DecisionTree(min_samples_split=3, max_depth=self.max_depth))
            self.trees[i].fit(X_subset, y_subset)
            
    def predict(self,X):
        predictions = []
        for row in self.trees:
            predictions.append(row.predict(X))
