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

    lambda_param: float
        Regularization parameter for the SVM.
        
    n_iters: int
        The number of iterations that the classifier will train over the dataset.

    Attributes:
    -----------
    w: 1d-array
        The weights after fitting. This is a 1d array of shape (n_features,).

    b: float
        The bias after fitting. If fit_intercept = False, the bias will be set to zero.

    Methods:
    --------
    fit(X, y): void
        Fit the model according to the given training data.

    predict(X): 1d-array
        Predict class labels for samples in X.

    accuracy(y_true, y_pred): float
        Returns the mean accuracy on the given test data and labels.

    score(X, y): float
        Returns the mean accuracy on the given test data and labels.

    plot(X, y): void
        Plot the decision boundary.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the SVM classifier.

        Parameters:
        -----------
        learning_rate: float
            The step length that will be taken when following the negative gradient during
            training.

        lambda_param: float
            Regularization parameter for the SVM.

        n_iters: int
            The number of iterations that the classifier will train over the dataset.

        Returns:
        --------
        None
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

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
        n_features = x.shape[1]
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(x):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * \
                        (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, x):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        x: 2d-array, shape=(n_samples, n_features)
            The training data.

        Returns:
        --------
        predictions: 1d-array, shape=(n_samples,)
            The predicted class labels.
        """
        approx = np.dot(x, self.w) - self.b
        return np.sign(approx)
    
    def plot(self, x, y):
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

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)

        x0_1 = np.amin(x[:, 0])
        x0_2 = np.amax(x[:, 0])

        x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
        x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, self.w, self.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.w, self.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, self.w, self.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.w, self.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

        x1_min = np.amin(x[:, 1])
        x1_max = np.amax(x[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()
    
    def accuracy(self, y_true, y_pred):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        y_true: 1d-array, shape=(n_samples,)
            The true labels.

        y_pred: 1d-array, shape=(n_samples,)
            The predicted labels.

        Returns:
        --------
        accuracy: float
            The mean accuracy of the classifier.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    def score(self, x, y):
        """
        Returns the mean accuracy on the given test data and labels.

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