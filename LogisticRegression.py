import numpy as np
from sklearn.linear_model import LogisticRegression as LogisticRegressionSklearnModel


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # Calculate the predicted values
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return y_pred

    def backward(self, X, y, y_pred):
        m = len(y)

        # Calculate gradients
        dz = y_pred - y
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)

        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.num_iterations):
            # Forward pass
            y_pred = self.forward(X)

            # Backward pass
            self.backward(X, y, y_pred)

    def predict(self, X):
        # Make predictions using the learned weights and bias
        z = np.dot(X, self.weights) + self.bias
        y_pred_prob = self.sigmoid(z)
        # Round the values to 0 or 1
        y_pred = np.round(y_pred_prob)
        return y_pred


class LogisticRegressionSklearn:
    def __init__(self):
        self.model = LogisticRegressionSklearnModel()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
