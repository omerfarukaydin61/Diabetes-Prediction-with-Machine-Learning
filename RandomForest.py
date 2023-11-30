from DecisionTree import DecisionTree
import numpy as np
import random

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Create n_estimators decision trees
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            # The order of the features doesnt matter so we can do it like this
            random_features = np.random.choice(X.shape[1], size=4, replace=False)
            X_subset = X[:, random_features]

            tree.fit(X_subset, y)

            self.trees.append(tree)

    def predict(self, X):
        # Predict the class labels for each tree and return the majority vote
        predictions = [tree.predict(X) for tree in self.trees]
        preds = []
        for i in range(len(predictions[0])):
            pred = [predictions[j][i] for j in range(len(predictions))]
            preds.append(max(set(pred), key=pred.count))
        return preds
