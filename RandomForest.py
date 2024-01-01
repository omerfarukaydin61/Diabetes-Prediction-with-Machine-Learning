from DecisionTree import DecisionTree
import numpy as np
import random
np.random.seed(42)

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=3, criteria = 'entropy', sample_size = 0.25):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.sample_size = sample_size
        self.criteria = criteria

    def fit(self, X, y):
        # n_estimators of decision trees
        for _ in range(self.n_estimators):

            n_samples = X.shape[0]
            # Randomly sampling sample_size% of the data with replacement
            n_samples = int(n_samples * self.sample_size)
            rows = np.random.choice(n_samples, n_samples, replace=True)
            a = X[rows]
            b = y[rows]

            tree = DecisionTree(max_depth=self.max_depth, criteria = self.criteria)

            tree.fit(a, b)
            
            self.trees.append(tree)

    def predict(self, X):
        # Predicting the class labels for each tree and return the majority vote
        predictions = [tree.predict(X) for tree in self.trees]
        preds = []
        for i in range(len(predictions[0])):
            pred = [predictions[j][i] for j in range(len(predictions))]
            preds.append(max(set(pred), key=pred.count))
        return preds
