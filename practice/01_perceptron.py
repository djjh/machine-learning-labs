# Implementation of the Perceptron described in "Python Machine Learning"
# on pages 18-32.

import numpy as np
from matplotlib import pyplot as pp
from matplotlib.colors import ListedColormap
import pandas as pd


## Load the Dataset.

iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)

## Inspect the Dataset.

# Print a couple of rows fromt he data set.
print("Sample of the Iris Dataset:")
print(iris_data.tail())
print()

# Display a scatter plot of sepal vs petal length
# for the setosa and versicolor flowers.
y = iris_data.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = iris_data.iloc[0:100, [0, 2]].values
pp.figure()
pp.scatter(
    X[:50, 0],
    X[:50, 1],
    color='red',
    marker='x',
    label='setosa')
pp.scatter(
    X[50:100, 0],
    X[50:100, 1],
    color='blue',
    marker='x',
    label='versicolor')
pp.xlabel('sepal length')
pp.ylabel('petal length')
pp.legend(loc='upper left')
pp.show()


## Build the model.

class Perceptron(object):

    """Perceptron classifier

    Parameters
    ----------
    learning_rate : float32
        Learning rate (between 0.0 and 1.0)
    iterations : int
        Passes over the training dataset.

    Attributes
    ----------
    weights : 1d array
        Weights after fitting.
    errors : list
        Number of misclassifications in every epoch.

    """

    def __init__(self, learning_rate=0.01, iterations=10):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        """Fit training data

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number
            of samples and n_features is the number of
            features.
        y : {array-like}, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        n_features =  X.shape[1]
        self.weights = np.zeros(1 + n_features)
        self.errors = []

        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

model = Perceptron(learning_rate=0.1, iterations=10)


## Train the model.

model.fit(X, y)


## Plot the accuracy over training.

pp.plot(
    range(1, len(model.errors) + 1),
    model.errors,
    marker='o')
pp.xlabel('Epochs')
pp.ylabel('Number of misclassifications')
pp.show()


## Plot

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup marker generation and color map.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    pp.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pp.xlim(xx1.min(), xx1.max())
    pp.ylim(xx2.min(), xx2.max())

    # Plot class samples.
    for idx, cl in enumerate(np.unique(y)):
        pp.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=cmap(idx),
            marker=markers[idx],
            label=cl)

plot_decision_regions(X, y, classifier=model)
pp.xlabel('sepal length [cm]')
pp.ylabel('petal length [cm]')
pp.legend(loc='upper left')
pp.show()
