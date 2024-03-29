import numpy as np

class LinearRegression:
    """
    This class implements the Linear Regression algorithm.

    Parameters:
    -----------
    learning_rate: float
        The learning rate of the model.

    regularization_param: float
        The regularization parameter of the model.

    epochs: int
        The number of iterations to train the the linear regression model.

    Attributes:
    -----------
    weights: ArrayLike
        The weights of the model.

    bias: float
        The bias of the model.

    Methods:
    --------
    compute_cost(x, y)
        This function computes the cost of the model.

    fit(x, y)
        This function fits the model to the data.

    predict(x)
        This function predicts the output of the model.

    r2_score(y_true, y_pred)
        This function computes the r2 score of the model.
    """
    
    def __init__(self, learning_rate=0.01, regularization_param=0, epochs=1000):
        """
        This function initializes the LinearRegression class.

        Parameters:
        -----------
        learning_rate: float
            The learning rate of the model.

        regularization_param: float
            The regularization parameter of the model.

        epochs: int
            The number of iterations to train the the linear regression model.

        Returns:
        --------
        None
        """
        
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.epochs = epochs
        self.weights = None
        self.bias = None

    
    def compute_cost(self, x, y):
        """
        This function computes the cost of the model.

        Parameters:
        -----------
        x: MatrixLike | ArrayLike,
            The input data of shape (m,n) where m is the number of samples and n is the number of features.

        y: ArrayLike
            The input data of shape (m,1) where m is the number of samples.
        
        Returns:
        --------
        The cost of the model.
        """

        n_samples = len(y)
        y_pred = np.dot(x, self.weights) + self.bias
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
        reg_cost = (self.regularization_param / (2 * n_samples)) * np.sum(self.weights ** 2)
        return (cost+reg_cost)
    
    def _gradient_descent_linear(self, x, y):
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
        y_pred = np.dot(x, self.weights) + self.bias
        err = y_pred - y
        dw = np.zeros(n_features) 
        for i in range(n_features):
            dw[i] = (err * x[:,i]).mean()
        db = err.mean()
        self.weights = (1 - self.learning_rate*(self.regularization_param/n_samples)) * self.weights - self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, x, y):
        """
        This function fits the model to the data.

        Parameters:
        -----------
        x: MatrixLike | ArrayLike,
            The input data of shape (m,n) where m is the number of samples and n is the number of features.

        y: ArrayLike
            The input data of shape (m,1) where m is the number of samples.
        
        Returns:
        --------
        The cost of the model.
        """

        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        cost_graph = []
        x = np.array(x)
        for i in range(self.epochs+1):
            self._gradient_descent_linear(x, y)
            cost = self.compute_cost(x, y)
            cost_graph.append(cost)
            if(i % 100 == 0):
                print(f'epoch: {i}, cost: {cost}')
        return cost_graph
    
    def predict(self, x):
        """
        This function predicts the output of the model.

        Parameters:
        -----------
        x: MatrixLike | ArrayLike,
            The input data of shape (m,n) where m is the number of samples and n is the number of features.

        Returns:
        --------
        The predicted output of the model.
        """
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred
    
    def r2_score(self,y_true, y_pred):
        """
        This function computes the r2 score of the model.

        Parameters:
        -----------
        y_true: ArrayLike
            The input data of shape (m,1) where m is the number of samples.

        y_pred: ArrayLike
            The input data of shape (m,1) where m is the number of samples.

        Returns:
        --------
        The r2 score of the model.
        """
            
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        return r2
    
    def mse(self, y_true, y_pred):
        """
        This function computes the mean squared error of the model.

        Parameters:
        -----------
        y_true: ArrayLike
            The input data of shape (m,1) where m is the number of samples.
            
        y_pred: ArrayLike
            The input data of shape (m,1) where m is the number of samples.

        Returns:
        --------
        The mean squared error of the model.
        """
        
        return np.mean((y_true - y_pred) ** 2)
    
    def __repr__(self) -> str:
        return f'LinearRegression(learning_rate={self.learning_rate}, regularization_param={self.regularization_param}, epochs={self.epochs})'
    
    def __str__(self) -> str:
        return self.__repr__()