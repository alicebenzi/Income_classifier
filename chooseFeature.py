import collections
import sklearn
import math
import numpy as np
import sys


class chooseFeature(sklearn.base.BaseEstimator):
    """
    This defines a classifier that predicts on the basis of
      the feature that was found to have the best weighted purity, based on splitting all
      features according to their mean value. Then, for that feature split, it predicts
      a new example based on the mean value of the chosen feature, and the majority class for
      that split.
      You can of course define more variables!
    """

    def __init__(self):
        # if we haven't been trained, always return 1
        self.classForGreater= 1
        self.classForLeq = 1
        self.chosenFeature = 0
        self.type = "chooseFeatureClf"
        self.mean = 0

    def impurity(self, labels):
        num_labels = len(labels)
        label_counter = collections.Counter(labels)
        label_count = label_counter.values()
        entropy = 0
        for i in range(0, len(label_count)):
            p_i = label_count[i]/float(num_labels)
            entropy += -p_i * math.log(p_i, 2)
        return entropy

    def weighted_impurity(self, list_of_label_lists):
        pass
        # ## TODO: Your code here, uses impurity
        total_num_labels = sum([len(label_list) for label_list in list_of_label_lists])
        weighted_entropy = 0
        for label_list in list_of_label_lists:
            weighted_entropy += len(label_list)/float(total_num_labels) * self.impurity(label_list)
        return weighted_entropy

    def ftr_seln(self, data, labels):
        """return: index of feature with best weighted_impurity, when split
        according to its mean value; you are permitted to return other values as well,
        as long as the the first value is the index
        """
        data = np.asarray(data)
        labels = np.asarray(labels)
        num_attributes = len(data[0])
        min_index = -1
        min_impurity = sys.maxint
        for i in range(0, num_attributes):
            x = data[:, i]
            mean_attribute = np.mean(x)
            y_greater = []
            y_lesser = []
            y_list = []
            for xi, yi in zip(x, labels):
                if xi > mean_attribute:
                    y_greater.append(yi)
                else:
                    y_lesser.append(yi)
            y_list.append(y_greater)
            y_list.append(y_lesser)
            weighted_impurity = self.weighted_impurity(y_list)
            if weighted_impurity < min_impurity:
                min_impurity = weighted_impurity
                min_index = i
        return min_index

    def fit(self, data, labels):
        """
        Inputs: data: a list of X vectors
        labels: Y, a list of target values
        """
        greater = []
        lesser = []
        index = self.ftr_seln(data, labels)
        self.chosenFeature = index
        mean = np.mean(data[:, index])

        x = data[:, index]
        for i in range(len(data[:, index])):
            if x[i] > mean:
                greater.append(labels[i])
            else:
                lesser.append(labels[i])
        greaterCounter = collections.Counter(greater)
        lesserCounter = collections.Counter(lesser)
        if len(greater) != 0:
            self.classForGreater = greaterCounter.most_common()[0][0]
        if len(lesser) != 0:
            self.classForLeq = lesserCounter.most_common()[0][0]
        self.mean = mean

    def predict(self, testData):
        """
        Input: testData: a list of X vectors to label.
        Check the chosen feature of each
        element of testData and make a classification decision based on it
        """
        # ## TODO: Your code here
        feature = testData[:, self.chosenFeature]
        return [self.classForGreater if x > self.mean else self.classForLeq for x in feature]



