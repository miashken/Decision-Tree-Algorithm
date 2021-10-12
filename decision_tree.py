import numpy as np
import pandas as pd
from sklearn import model_selection

class Leaf:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def predict(self, row):
        return self.get_value()

class Node:
    def __init__(self, level, split_feature, split_value, left_node=None, right_node=None):
        self.level = level
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node

    def predict(self, row):
        if row[self.split_feature] >= self.split_value:
            return self.right_node.predict(row)
        return self.left_node.predict(row)

class DecisionTreeClassifier:

    def __init__(self, max_depth, gini=True):
        self.max_depth = max_depth
        self.gini = gini
        self.root = None

    def set_root(self, node):
        if self.root == None:
            self.root = node

    def calc_popular_class(self, y):
        values, freq = np.unique(y, return_counts=True)
        idx = np.argmax(freq)
        popular_class = values[idx]
        return popular_class

    def calc_metrics(self, y):
        '''
        This function calc gini score (or entropy), to measure how accurate the split was.

        gini = 1 - sum(p^2)
        entropy = sum(p * log(p))
        '''
        values, freq = np.unique(y, return_counts=True)
        branch_size = float(len(y))
        class_probabilities = freq / branch_size
        if self.gini:
            metric = 1 - np.sum(class_probabilities**2, axis=0)
        else:
            metric = np.sum(class_probabilities * np.log(class_probabilities), axis=0)
        return metric

    def get_best_split(self, X, y):
        num_features = X.shape[1]
        num_rows = len(y)
        best_split_feature = 0
        best_split_value = 0
        best_score = np.inf

        for feature in range(num_features - 1):
            values = np.unique(X[:,feature])
            for val in values:
                #split data for specific value and feature
                right_data, right_labels, left_data, left_labels = self.data_split(X, y, feature, val)

                #calc score (check if the split was good)
                score = 0
                for subtree in [right_labels, left_labels]:
                    score += len(subtree) * self.calc_metrics(subtree) / num_rows

                if score < best_score:
                    best_score = score
                    best_split_feature, best_split_value = feature, val

        return best_split_feature, best_split_value, best_score

    def data_split(self, X, y, split_feature, split_value):
        # -- right --
        idx_right_subtree = X[:, split_feature] >= split_value
        right_subtree = X[idx_right_subtree]
        right_subtree_labels = y[idx_right_subtree]

        # -- left --
        idx_left_subtree = X[:, split_feature] < split_value
        left_subtree = X[idx_left_subtree]
        left_subtree_labels = y[idx_left_subtree]

        return right_subtree, right_subtree_labels, left_subtree, left_subtree_labels

    def fit(self, X, y):
        ''' building the tree '''
        self.set_root(self.split_node(X,y))

    def split_node(self, X, y, node_level=0):
        node_level +=1

        if len(y) == 1:
            return Leaf(y[0])

        split_feature, split_value, score = self.get_best_split(X, y)
        if score == 0.0 or self.max_depth < node_level:
            popular_class = self.calc_popular_class(y)
            return Leaf(popular_class)

        right_subtree, right_subtree_labels, left_subtree, left_subtree_labels = self.data_split(X, y, split_feature, split_value)
        if len(right_subtree_labels) == 1:
            return Leaf(right_subtree_labels[0])
        if len(left_subtree_labels) == 1:
            return Leaf(left_subtree_labels[0])

        right_node = self.split_node(right_subtree,right_subtree_labels, node_level)
        left_node = self.split_node(left_subtree, left_subtree_labels, node_level)

        return Node(node_level, split_feature, split_value, left_node, right_node)

    def predict_labels(self, X_test):
        preds = []
        for row in X_test:
            preds.append(self.root.predict(row))
        return np.asarray(preds)

    def get_accuracy(self, y, preds):
        correct = y == preds
        acc = ((np.sum(correct) / len(y)) * 100.0).round(2)
        return acc

if __name__ == '__main__':
    # -- Data --
    DATA_PATH = './wdbc.data'
    breast_cancer_data = pd.read_csv(DATA_PATH, header=None)
    X = np.asarray(breast_cancer_data.iloc[:, 2:])
    y = np.asarray(breast_cancer_data.iloc[:,1]).astype('str')

    # -- train, test split --
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.25)

    # -- define and train the model --
    Max_DEPTHS = [4, 8, 16]
    for max_depth in Max_DEPTHS:
        tree = DecisionTreeClassifier(max_depth)
        tree.fit(X_train, y_train)
        preds = tree.predict_labels(X_test)
        acc = tree.get_accuracy(y_test, preds)
        print('for max depth of: ', max_depth, ' accuracy is: ', acc)

