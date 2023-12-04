import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, activation="relu", optimizer="adam", verbose=True):
        super(MultilayerPerceptron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.verbose = verbose

        # Create the linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        # Create the activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

        # Create the optimizer
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer")

    def forward(self, X):
        # Calculate the output values
        out = self.linear1(X)
        out = self.activation(out)
        out = self.linear2(out)
        return out

    def loss(self, y_pred, y_true):
        # Calculate and return the loss value
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y_true)
        return loss

    def fit(self, X, y, epochs):
        self.loss_values = []
        self.accuracy_values = []
        # Train the MLP on the given data for the given number of epochs
        # Loop over the epochs
        for epoch in range(epochs):
            # Zero the gradients
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.forward(X)
            # Calculate the loss
            loss = self.loss(y_pred, y)
            # Backward pass
            loss.backward()
            self.loss_values.append(loss.item())
            accuracy = torch.mean(torch.eq(y_pred.argmax(dim=1), y).float())
            self.accuracy_values.append(accuracy.item())
            # Update the parameters
            self.optimizer.step()
            # Print the loss value every 10 epochs
            if epoch % 100 == 0 and self.verbose:
                print(f"Epoch {epoch}, \033[91mTraining Loss: {loss.item()}\033[0m - \033[92mTraining Accuracy: {accuracy.item()}\033[0m")
        if self.verbose:
            plt.plot(self.loss_values)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")

    def predict(self, X):
        # Make predictions using the trained MLP
        # Forward pass
        y_pred = self.forward(X)
        return y_pred
