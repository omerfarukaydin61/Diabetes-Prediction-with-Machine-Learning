import math
import numpy as np
import pandas as pd


class Node():
    def __init__(self, feature=None, left=None, right=None, information_gain=None, value=None, data=None):
        self.feature = feature
        self.data = data
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.value = value

class DecisionTree():

    def __init__(self, min_samples_split=2,max_depth = 2, criteria = 'entropy'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.criteria = criteria

    def entropy(self,y_values):
        # Data is numpy array
        
        entropy = 0
        # our labels are either 1 or 0
        no_of_labels = 2

        # Entropy formula = -sum(p * log(p))
        for i in range(no_of_labels):
            p = len(y_values[y_values==i]) / len(y_values)
            if p != 0:
                entropy += -p * math.log(p, 2)
            else:
                entropy += 0
        return entropy
    
    def gini_index(self, y_values):
        # Gini formula = 1 - sum(p^2)
        gini = 1
        no_of_labels = len(np.unique(y_values))

        for i in range(no_of_labels):
            p = len(y_values[y_values == i]) / len(y_values)
            gini -= p ** 2

        return gini


    def information_gain(self,parent,left,right):

        # IG formula = entropy(parent) - (w)* entropy(children)
        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)

        if self.criteria == "gini":
            return self.gini_index(parent) - (left_weight * self.gini_index(left) + right_weight * self.gini_index(right))
        else:
            # Default is entropy
            return self.entropy(parent) - (left_weight * self.entropy(left) + right_weight * self.entropy(right))


    def fit(self, X, y):
        # Concatenating the X and y values
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.create_tree(dataset, max_depth=self.max_depth)

    
    def best_split(self, dataset):

        best_split = {'feature': None, 'value': None, 'left': None, 'right': None, 'info_gain': -1000}
        best_info_gain = -1000
        # Looping through all the features
        for feature in range(dataset.shape[1]-1):
            # Looping through all the unique values of the feature
            for value in np.unique(dataset[:, feature]):
                left_data = np.array([i for i in dataset if i[feature] <= value])
                right_data = np.array([i for i in dataset if i[feature] > value])

                # If length of left and right data is greater than 0 we calculate the information gain
                if len(left_data) > 0 and len(right_data) > 0:
                    y = dataset[:, -1]
                    left_y = left_data[:, -1]
                    right_y = right_data[:, -1]
                    info_gain = self.information_gain(y, left_y, right_y)

                    # If the calculated information gain is greater than the best information gain then we update the best split
                    if info_gain > best_info_gain:
                        best_split['feature'] = feature
                        best_split['value'] = value
                        best_split['left'] = left_data
                        best_split['right'] = right_data
                        best_split['info_gain'] = info_gain
                        best_info_gain = info_gain
        return best_split


    def create_tree(self,dataset,counter=0,max_depth=2):

        # If the max depth is reached or the number of samples in the dataset is less than the minimum samples
        # required to split then we return the leaf node with the majority class
        if max_depth >= counter and len(dataset[:, -1]) >= self.min_samples_split:

            best_split = self.best_split(dataset)

            if best_split['info_gain'] > 0:
                left = self.create_tree(best_split['left'], counter + 1, self.max_depth)
                right = self.create_tree(best_split['right'], counter + 1, self.max_depth)
                return Node(best_split['feature'], left, right, best_split['info_gain'],
                             best_split['value'])
            
        # Returning the leaf node with the majority class if threshold is reached
        Y = dataset[:, -1]
        leaf_value = max(Y, key=list(Y).count)
        return Node(data=leaf_value)
    
    # predicting by traversing the tree
    def predictions(self, X,tree):
        if tree.data != None:
            return tree.data
        feature_value = X[tree.feature]
        if feature_value <= tree.value:
            return self.predictions(X,tree.left)
        else:
            return self.predictions(X,tree.right)
        
    # Predicting for each row in the dataset
    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self.predictions(row,self.root))
        return predictions
    
    # Confusion Matrix
    def confusion_matrix(self,y_true, y_pred):
        # Creating a confusion matrix
        confusion_matrix = np.zeros((2, 2), dtype=int)

        # 0,0 = TP  0,1 = FN 1,0 = FP  1,1 = TN
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i] == 0:
                confusion_matrix[1][1] += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                confusion_matrix[1][0] += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                confusion_matrix[0][1] += 1
            elif y_true[i] == y_pred[i] == 1:
                confusion_matrix[0][0] += 1

        return confusion_matrix
    
    # Evaluation Metrics accuracy, precision, recall, f1 score
    def metrics(self,confusion_matrix):
        # Accuracy = TP + TN / TP + TN + FP + FN
        accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (
                    confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[0][1] + confusion_matrix[1][0])

        # Precision = TP / TP + FP
        precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])

        # Recall = TP / TP + FN
        recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return accuracy, precision, recall, f1_score
