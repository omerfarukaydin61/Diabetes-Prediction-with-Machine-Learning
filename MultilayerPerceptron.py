import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation="relu", optimizer="adam"):
        super(MultilayerPerceptron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.optimizer = optimizer

        # Create the linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        # Create the activation function
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

        # Create the optimizer
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        else:
            raise ValueError("Invalid optimizer")

    def forward(self, X):
        # Calculate the output values
        out = self.linear1(X)
        out = self.act(out)
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
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, \033[91mTraining Loss: {loss.item()}\033[0m - \033[92mTraining Accuracy: {accuracy.item()}\033[0m")

        plt.plot(self.loss_values)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    def predict(self, X):
        # Make predictions using the trained MLP
        # Forward pass
        y_pred = self.forward(X)
        return y_pred

    def confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def metrics(self, confusion_matrix):
        tn, fp, fn, tp = confusion_matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1_score
