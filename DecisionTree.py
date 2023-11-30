import math
import numpy as np
import pandas as pd


class Node():
    def __init__(self, feature, left,right, information_gain, entropy, data):
        self.feature = feature
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.entropy = entropy
        self.data = data


class DecisionTree():

    def __init__(self, min_samples_split=2,max_depth = 5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def entropy(self,y_values):
        # Data is numpy array
        
        entropy = 0
        # our labels are either 1 or 0
        no_of_labels = 2

        for i in range(no_of_labels):
            p = len(y_values[y_values==i]) / len(y_values)
            entropy += -p * math.log(p, 2)

        return entropy

    def information_gain(self,parent,left,right):

        # IG formula = entropy(parent) - (w)* entropy(children)
        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)

        return self.entropy(parent) - (left_weight * self.entropy(left) + right_weight * self.entropy(right))


        