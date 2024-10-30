import numpy as np
from utility_service import sigmoid


class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_iter=100000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        weights = np.zeros(np.shape(X)[1] + 1, dtype=np.float64)
        X = np.c_[np.ones((np.shape(X)[0], 1)), X]

        for iter in range(self.max_iter):
            z = np.dot(X, weights)
            y_pred = sigmoid(z)
            errors = y_pred - y
            gradients = np.dot(X.T, errors)
            weights = weights - self.learning_rate * gradients

        self.weights = weights

    def predict(self, X):
        X = np.c_[np.ones((np.shape(X)[0], 1)), X]
        z = np.dot(X, self.weights)
        y_pred = sigmoid(z)
        predictions = [1 if i > 0.5 else 0 for i in y_pred]
        return predictions