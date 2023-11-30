import numpy as np

class LogisticRegression:
    """
    Logistic Regression Classifier

    Parameters:
    -----------
    learning_rate: float
        The learning rate of the model.
    regularization_param: float
        The regularization parameter of the model.
    epochs: int
        The number of iterations to train the logistic regression model.
    threshold: float
        The decision threshold that separates the two classes 0 and 1.

    Attributes:
    -----------
    weights: ArrayLike
        The weights vector of the logistic regression model.
    bias: float
        The bias term of the logistic regression model.
    
    Methods:
    --------
    fit(x, y)
        This function performs the gradient descent algorithm.
    predict(x)
        This function predicts the class labels.
    compute_cost(x, y)
        This function calculates the cost function.
    accuracy(y_true, y_pred)
        This function calculates the accuracy of the model.

    """
    
    def __init__(self, learning_rate=0.01, regularization_param=0, epochs=10000, threshold=0.5):
        """
        This function initializes the logistic regression model.

        Parameters:
        -----------
        learning_rate: float
            The learning rate of the model.
        regularization_param: float
            The regularization parameter of the model.
        epochs: int
            The number of iterations to train the logistic regression model.
        threshold: float
            The decision threshold that separates the two classes 0 and 1.
        
        Returns:
        --------
        None
        """

        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None

    
    def _sigmoid(self, z):
        """
        This function calculates the Sigmoid of the input 'z'.

        Parameters:
        -----------
        z: float
            The input value to the sigmoid function.
        
        Returns:
        --------
        The sigmoid of the input 'z'.
        """

        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, x, y):
        """
        This function calculates the cost function.

        Parameters:
        -----------
        x: MatrixLike | ArrayLike,
            The input data of shape (m,n) where m is the number of samples and n is the number of features.
        y: ArrayLike
            The input data of shape (m,1) where m is the number of samples.
        
        Returns:
        --------
        cost: float
            The cost function value.
        """

        n_samples = len(y)
        y_pred = self._sigmoid(np.dot(x, self.weights) + self.bias)
        cost = (-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).mean()
        reg_cost = (self.regularization_param / (2 * n_samples)) * np.sum(self.weights ** 2)
        return (cost+reg_cost)
    
    def _gradient_descent_logistic(self, x, y):
        """
        This function performs the gradient descent algorithm.

        Parameters:
        -----------
        x: MatrixLike | ArrayLike,
            The input data of shape (m,n) where m is the number of samples and n is the number of features.
        y: ArrayLike
            The input data of shape (m,1) where m is the number of samples.
        
        Returns:
        --------
        None
        """
        
        n_samples, n_features = x.shape
        y_pred = self._sigmoid(np.dot(x, self.weights) + self.bias) 
        err = y_pred - y
        dw = np.zeros(n_features) 
        for i in range(n_features):
            dw[i] = (err * x[:,i]).mean()
        db = err.mean()
        self.weights = (1 - self.learning_rate*(self.regularization_param/n_samples)) * self.weights - self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, x, y):
        """
        This function fits the logistic regression model to the training data.

        Parameters:
        -----------
        x: MatrixLike | ArrayLike,
            The input data of shape (m,n) where m is the number of samples and n is the number of features.
        y: ArrayLike
            The input data of shape (m,1) where m is the number of samples.

        Returns:
        --------
        self
        """

        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        x = np.array(x)
        for i in range(self.epochs):
            self._gradient_descent_logistic(x, y)
            cost = self.compute_cost(x, y)
            if(i % 1000 == 0):
                print(f'epoch: {i}, cost: {cost}')
        return self
    
    def predict(self, x):
        """
        This function predicts the class labels.

        Parameters:
        -----------
        x: MatrixLike | ArrayLike,
            The input data of shape (m,n) where m is the number of samples and n is the number of features.
        
        Returns:
        --------
        The predicted class labels.
        """

        y_pred = self._sigmoid(np.dot(x, self.weights) + self.bias)
        return np.where(y_pred > self.threshold, 1, 0)
    
    def accuracy(self,y_true, y_pred):
        """
        This function calculates the accuracy of the model.

        Parameters:
        -----------
        y_true: ArrayLike
            The input data of shape (m,1) where m is the number of samples.
        y_pred: ArrayLike
            The input data of shape (m,1) where m is the number of samples.

        Returns:
        --------
        The accuracy of the model.
        """
        
        return np.sum(y_true == y_pred) / len(y_true)