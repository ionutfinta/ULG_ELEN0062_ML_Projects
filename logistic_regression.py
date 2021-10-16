"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        #X = np.hstack((X, np.ones((X.shape[0], 1))))
        y = y.reshape((y.shape[0], 1))

        #-------------------- Modele
        self.sigmoid = lambda x: 1 / (1+np.exp(-x))
        self.init = lambda x: (np.random.randn(x.shape[1],1), np.random.randn(1))
        self.forward_propagation = lambda x, w, b: self.sigmoid(x.dot(w)+b)
        self.log_loss = lambda y, a: 1 / len(y) * np.sum(-y * np.log(a) - (1-y) * np.log(1-a))
        self.gradients = lambda x, a, y: (1/len(y) * np.dot(x.T, a-y), 1/len(y) * np.sum(a - y))


        # Optimisation (Gradient descent)
        def optimisation(X, W, b, A, y):
            dW, db = self.gradients(X, A, y)
            W = W - self.learning_rate * dW
            b = b - self.learning_rate * db
            return (W,b)

        # Initialisation
        self.W, self.b = self.init(X)
        self.loss_history = []

        # Training
        for i in range(self.n_iter):
            A = self.forward_propagation(X, self.W, self.b)
            self.loss_history.append(self.log_loss(y, A))
            self.W, self.b = optimisation(X, self.W, self.b, A, y)

        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        y= self.forward_propagation(X, self.W, self.b)>=0.5
        y = y.reshape(y.shape[0])
        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        y= self.forward_propagation(X, self.W, self.b)
        y = y.reshape(y.shape[0])
        return y

if __name__ == "__main__":
    pass
