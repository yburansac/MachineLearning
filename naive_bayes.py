# sources of inspiration:
# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

from math import sqrt
from math import exp
from math import pi
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np



class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be
    Gaussian.
    """
    def __init__(self):
        self._eps = 0.0000001

    # -----------------------API----------------------
    def fit(self, X, y):
        """
        Fit (train) the model according to data.
        :param X: training feature vectors
        :param y: target labels
        :return: self
        """
        feature_vector_width = len(X[0])
        running_sums = defaultdict(lambda: np.zeros(feature_vector_width + 1))

        # calculate means
        for feature_vector, label in zip(X, y):
            for i in range(feature_vector_width):
                running_sums[label][i] += feature_vector[i]
            running_sums[label][-1] += 1

        means = {
            label: [feature / feature_vector[-1] for feature in feature_vector[:-1]]
            for label, feature_vector in running_sums.items()
        }

        # calculate standard deviation
        for feature_vector in running_sums.values():
            feature_vector[:-1].fill(0)  # zero out the vectors except the last position with class sample counts

        for feature_vector, label in zip(X, y):
            for i in range(len(feature_vector)):
                running_sums[label][i] += (feature_vector[i] - means[label][i]) ** 2

        std_devs = {  # a small value is added to deal with zero variance features
            label: [sqrt(feature / feature_vector[-1]) + self._eps for feature in feature_vector[:-1]]
            for label, feature_vector in running_sums.items()
        }

        # calculate prior probabilities
        cls_prior_probs = {label: feature_vector[-1] / len(X)
                           for label, feature_vector in running_sums.items()
                           }
        self._classes = sorted(means.keys())
        self._means = means
        self._std_devs = std_devs
        self._cls_prior_probs = cls_prior_probs
        return self

    def predict(self, X):
        """
        Classify data according to model.
        :param X: feature vectors to classify
        :return: predicted labels for X
        """
        predictions = np.empty(shape=len(X), dtype=int)
        for i in range(len(X)):
            max_score = 0
            max_score_class = 0
            for cls in self._classes:
                score = self._cls_prior_probs[cls]
                for feature_index in range(len(X[i])):
                    score *= self._gauss_normal_PDF(X[i][feature_index], self._means[cls][feature_index],
                                                    self._std_devs[cls][feature_index])
                if score > max_score:
                    max_score = score
                    max_score_class = cls
            predictions[i] = max_score_class
        return predictions

        # -----------------------Internal-----------------

    @staticmethod
    def _gauss_normal_PDF(x, mean, stdev):
        """
        Calculate the value of probability density function of normal distribution.
        # TODO consider deleting
        # this is a substitute for norm.pdf(x, loc, scale) from scipy.stats,
        # it is not needed and can be deleted, probably slower
        :param x: x
        :param mean: the mean of the distribution
        :param stdev: the standard deviation of the distribution
        :return: the PDF of the distribution at x
        """
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
