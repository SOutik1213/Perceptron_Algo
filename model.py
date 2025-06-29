import pandas as pd
import numpy as np

class Perceptron(object):
    def __init__(self, lr=0.001, max_iter=1000, eps=1e-5):
        self.learning_rate = lr
        self.max_iter = max_iter
        self.eps = eps
        self.theta = None

    def add_intercept(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten()

        X = self.add_intercept(X)
        m, n = X.shape

        assert y.shape[0] == m, f"Mismatch: X has {m} samples, y has {y.shape[0]} labels"

        self.theta = np.zeros(n)

        for _ in range(self.max_iter):
            theta_old = self.theta.copy()
            hx = X @ self.theta
            gz = np.where(hx >= 0, 1, 0)
            gradient = (y - gz) @ X
            self.theta += self.learning_rate * gradient

            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.add_intercept(X)
        hx = X @ self.theta
        return np.where(hx >= 0, 1, 0)
