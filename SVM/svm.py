import numpy as np
import matplotlib.pyplot as plt

class SVM:
    """
    Support Vector Machine classifier.

    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.

    regularization_param: float
        Regularization parameter for the SVM.
        
    num_iterations: int
        The number of iterations that the classifier will train over the dataset.

    Attributes:
    -----------
    weights: 1d-array
        The weights after fitting. This is a 1d array of shape (n_features,).

    bias: float
        The bias after fitting. If fit_intercept = False, the bias will be set to zero.

    Methods:
    --------
    fit(x, y): void
        Fit the model according to the given training data.

    predict(x): 1d-array
        Predict class y for samples in x.

    accuracy(y_true, y_pred): float
        Returns the mean accuracy on the given test data and y.

    score(x, y): float
        Returns the mean accuracy on the given test data and y.

    plot_decision_boundary(x, y): void
        Plot the decision boundary.
    """
    def __init__(self, learning_rate=0.001, regularization_param=0.01, num_iterations=1000):
        """
        Initialize the SVM classifier.

        Parameters:
        -----------
        learning_rate: float
            The step length that will be taken when following the negative gradient during
            training.

        regularization_param: float
            Regularization parameter for the SVM.

        num_iterations: int
            The number of iterations that the classifier will train over the dataset.

        Returns:
        --------
        None
        """
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        """
        Fit the model according to the given training data.

        Parameters:
        -----------
        x: 2d-array, shape=(n_samples, n_features)
            The training data.

        y: 1d-array, shape=(n_samples,)
            The target values.

        Returns:
        --------
        None
        """
        y_transformed = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            for xi, yi in zip(x, y_transformed):
                is_correct_classification = yi * (np.dot(xi, self.weights) - self.bias) >= 1
                self.weights -= self.learning_rate * (2 * self.regularization_param * self.weights) if is_correct_classification else self.learning_rate * (2 * self.regularization_param * self.weights - np.dot(xi, yi))
                self.bias -= self.learning_rate * yi if not is_correct_classification else 0

    def predict(self, x):
        """
        Predict class y for samples in x.

        Parameters:
        -----------
        x: 2d-array, shape=(n_samples, n_features)
            The training data.

        Returns:
        --------
        predictions: 1d-array, shape=(n_samples,)
            The predicted class y.
        """
        approximations = np.dot(x, self.weights) - self.bias
        return np.sign(approximations)

    def plot_decision_boundary(self, x, y):
        """
        Plot the decision boundary.

        Parameters:
        -----------
        x: 2d-array, shape=(n_samples, n_features)
            The training data.

        y: 1d-array, shape=(n_samples,)
            The target values.

        Returns:
        --------
        None
        """
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        _, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1], marker='o', c=y)

        x_min, x_max = np.amin(x[:, 0]), np.amax(x[:, 0])
        x1_min, x1_max = get_hyperplane_value(x_min, self.weights, self.bias, 0), get_hyperplane_value(x_max, self.weights, self.bias, 0)
        x1_min_m, x1_max_m = get_hyperplane_value(x_min, self.weights, self.bias, -1), get_hyperplane_value(x_max, self.weights, self.bias, -1)
        x1_min_p, x1_max_p = get_hyperplane_value(x_min, self.weights, self.bias, 1), get_hyperplane_value(x_max, self.weights, self.bias, 1)

        ax.plot([x_min, x_max], [x1_min, x1_max], 'y--')
        ax.plot([x_min, x_max], [x1_min_m, x1_max_m], 'k')
        ax.plot([x_min, x_max], [x1_min_p, x1_max_p], 'k')

        x1_min, x1_max = np.amin(x[:, 1]) - 3, np.amax(x[:, 1]) + 3
        ax.set_ylim([x1_min, x1_max])

        plt.show()

    def accuracy(self, y_true, y_pred):
        """
        Returns the mean accuracy on the given test data and y.

        Parameters:
        -----------
        y_true: 1d-array, shape=(n_samples,)
            The true y.

        y_pred: 1d-array, shape=(n_samples,)
            The predicted y.

        Returns:
        --------
        accuracy: float
            The mean accuracy of the classifier.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def score(self, x, y):
        """
        Returns the mean accuracy on the given test data and y.

        Parameters:
        -----------
        x: 2d-array, shape=(n_samples, n_features)
            The training data.

        y: 1d-array, shape=(n_samples,)
            The target values.

        Returns:
        --------
        score: float
            The mean accuracy of the classifier.
        """
        pred = self.predict(x)
        return self.accuracy(y, pred)

    def __str__(self):
        return "SVM Classifier"

    def __repr__(self):
        return self.__str__()