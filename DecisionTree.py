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


class DecisionTree(object):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.tree = self.build_decision_tree(data, labels)

    def entropy(self,choice1,choice2):
        
        choice1_entropies = []
        choice2_entropies = []


        if len(choice1) != 0:
            for i in range(len(choice1)):
                label1 = choice1[i].count(1)
                label0 = choice1[i].count(0)
                if label1 != 0 and label0 != 0:
                    choice1_entropies.append(-((label1/len(choice1[i]))*math.log2(label1/len(choice1[i])))-((label0/len(choice1[i]))*math.log2(label0/len(choice1[i]))))
                else:
                    choice1_entropies.append(0)
        if len(choice2) != 0:
            for i in range(len(choice2)):
                label1 = choice2[i].count(1)
                label0 = choice2[i].count(0)
                if label1 != 0 and label0 != 0:
                    choice2_entropies.append(-((label1/len(choice2[i]))*math.log2(label1/len(choice2[i])))-((label0/len(choice2[i]))*math.log2(label0/len(choice2[i]))))
                else:
                    choice2_entropies.append(0)

        return choice1_entropies, choice2_entropies
    
    def information_gain(self,parent,children):

        info_gain = 0

        if len(children) != 0:
            for i in range(len(children)):
                info_gain += (len(children[i])/len(parent))*self.entropy([children[i]],[children[i]])[0][0]
        

        info_gain = self.entropy([parent],[parent])[0][0] - info_gain

        return info_gain




        