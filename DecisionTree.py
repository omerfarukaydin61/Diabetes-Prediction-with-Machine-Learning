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

    def __init__(self, min_samples_split=2,max_depth = 2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def entropy(self,y_values):
        # Data is numpy array
        
        entropy = 0
        # our labels are either 1 or 0
        no_of_labels = 2

        for i in range(no_of_labels):
            p = len(y_values[y_values==i]) / len(y_values)
            if p != 0:
                entropy += -p * math.log(p, 2)
            else:
                entropy += 0
        return entropy


    def information_gain(self,parent,left,right):

        # IG formula = entropy(parent) - (w)* entropy(children)
        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)

        return self.entropy(parent) - (left_weight * self.entropy(left) + right_weight * self.entropy(right))


    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.create_tree(dataset)

    
    def best_split(self, dataset):

        best_split = {'feature': None, 'value': None, 'left': None, 'right': None, 'info_gain': -1000}
        best_info_gain = -1000
        # Loop through all the features
        for feature in range(dataset.shape[1]-1):
            # Loop through all the unique values of the feature
            for value in np.unique(dataset[:, feature]):
                left_data = np.array([i for i in dataset if i[feature] <= value])
                right_data = np.array([i for i in dataset if i[feature] > value])

                if len(left_data) > 0 and len(right_data) > 0:
                    y = dataset[:, -1]
                    left_y = left_data[:, -1]
                    right_y = right_data[:, -1]
                    info_gain = self.information_gain(y, left_y, right_y)

                    if info_gain > best_info_gain:
                        best_split['feature'] = feature
                        best_split['value'] = value
                        best_split['left'] = left_data
                        best_split['right'] = right_data
                        best_split['info_gain'] = info_gain
                        best_info_gain = info_gain
        return best_split


    def create_tree(self,dataset,counter=0,max_depth=2):

        if max_depth >= counter and len(dataset[:, -1]) >= self.min_samples_split:

            best_split = self.best_split(dataset)

            if best_split['info_gain'] > 0:
                left = self.create_tree(best_split['left'], counter + 1, self.max_depth)
                right = self.create_tree(best_split['right'], counter + 1, self.max_depth)
                return Node(best_split['feature'], left, right, best_split['info_gain'],
                             best_split['value'])
            
            # Return the leaf node with the majority class if threshold is reached
        Y = dataset[:, -1]
        leaf_value = max(Y, key=list(Y).count)
        return Node(data=leaf_value)
    

    def predictions(self, X,tree):
        if tree.data != None:
            return tree.data
        feature_value = X[tree.feature]
        if feature_value <= tree.value:
            return self.predictions(X,tree.left)
        else:
            return self.predictions(X,tree.right)
    
    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self.predictions(row,self.root))
        return predictions
    

    def confusion_matrix(self,y_true, y_pred):
        # create a confusion matrix
        confusion_matrix = np.zeros((2, 2), dtype=int)

        # 1,1 = True Positive 0,0 = True Negative 0,1 = False Positive 1,0 = False Negative
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                confusion_matrix[0][0] += 1
            if y_pred[i] == 1 and y_true[i] == 0:
                confusion_matrix[0][1] += 1
            if y_pred[i] == 0 and y_true[i] == 1:
                confusion_matrix[1][0] += 1
            if y_pred[i] == 0 and y_true[i] == 0:
                confusion_matrix[1][1] += 1

        return confusion_matrix
    
    # Evaluation Metrics accuracy, precision, recall, f1 score
    def metrics(self,confusion_matrix):
        # Accuracy = TP + TN / TP + TN + FP + FN
        accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (
                    confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[0][1] + confusion_matrix[1][0])

        # Precision = TP / TP + FP
        precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])

        # Recall = TP / TP + FN
        recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return accuracy, precision, recall, f1_score
