import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Create the activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activation,
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        )

        # Create the optimizer
        if optimizer == "adam":
            self.optimizer = optim.Adam(
                self.parameters(), lr=self.learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer")

    def forward(self, X):
        # Forward pass
        return self.model(X)

    def loss(self, y_pred, y_true):
        # Calculate the loss
        return self.criterion(y_pred, y_true)

    def fit(self, X, y, epochs):
        # Train the MLP
        self.loss_values = []
        self.accuracy_values = []
        # Train the MLP for the specified number of epochs
        for epoch in range(epochs):
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
            self.optimizer.step()
            # Print the loss and accuracy at every 100th epoch
            if epoch % 100 == 0 and self.verbose:
                print(
                    f"Epoch {epoch}, \033[91mTraining Loss: {loss.item()}\033[0m - \033[92mTraining Accuracy: {accuracy.item()}\033[0m")
        # Plot the training curve
        if self.verbose:
            self.plot_training_curve()

    def plot_training_curve(self):
        plt.plot(self.loss_values, label="Loss")
        plt.plot(self.accuracy_values, label="Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.show()

    def predict(self, X):
        # Make predictions using the trained MLP
        # Forward pass
        y_pred = self.forward(X)
        return y_pred
