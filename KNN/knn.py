import numpy as np

class KNN:
    """
    This class implements the KNN algorithm.

    Parameters:
    -----------
    k: int
        The number of nearest neighbors.

    type: str
        The type of the model. It can be either 'classification' or 'regression'.
    
    Attributes:
    -----------
    x: ArrayLike
        The features of the model.

    y: ArrayLike
        The labels of the model.

    Methods:
    --------
    fit(x, y)
        This function fits the model to the data.

    predict(x)
        This function predicts the output of the model.

    accuracy(y_true, y_pred)
        This function computes the accuracy of the model.

    score(x, y)
        This function computes the accuracy of the model.
    """
    def __init__(self, k=5, type='classification'):
        """
        This function initializes the KNN class.

        Parameters:
        -----------
        k: int
            The number of nearest neighbors.

        type: str
            The type of the model. It can be either 'classification' or 'regression'.

        Returns:
        --------
        None
        """
        self.k = k
        self.type = type
        self.x = None
        self.y = None
    
    def fit(self, x, y):
        """
        Fit KNN classifier

        Parameters:
        -----------
        x: array-like, shape (n_samples, n_features)
            The training input samples.

        y: array-like, shape (n_samples, )
            The target values.

        Returns:
        --------
        self: object
        """
        self.x = x
        self.y = y
        return self
    
    def _predict_classification(self, x):
        """
        Predict class labels for samples in x using a majority vote.

        Parameters:
        -----------
        x: array-like, shape (n_samples, n_features)
            The training input samples.

        Returns:
        --------
        y_pred: array-like, shape (n_samples, )
            The target values.
        """

        y_pred = np.empty(x.shape[0], dtype=self.y.dtype)
        for i in range(x.shape[0]):
            distances = np.sum((self.x - x[i]) ** 2, axis=1)
            min_dist_ind = np.argsort(distances)[:self.k]
            y_min_dist = self.y[min_dist_ind]
            y_pred[i] = np.argmax(np.bincount(y_min_dist))
        return y_pred
    
    def _predict_regression(self, x):
        """
        Predict numeric values for samples using a mean of k nearest neighbors.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y_pred: array-like, shape (n_samples, )
            The target values.
        """
        y_pred = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            distances = np.sum((self.x - x[i]) ** 2, axis=1)
            min_dist_ind = np.argsort(distances)[:self.k]
            y_min_dist = self.y[min_dist_ind]
            y_pred[i] = np.mean(y_min_dist)
        return y_pred
    
    def predict(self, x):
        """
        Predict the output of the model.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_features)
            The training input samples.


        Returns
        -------
        y: array-like, shape (n_samples, )
            The target values.
        """
        if self.type == 'classification':
            return self._predict_classification(x)
        elif self.type == 'regression':
            return self._predict_regression(x)
        else:
            raise ValueError('type must be either classification or regression')
    
    def _get_accuracy_classification(self, y_true, y_pred):
        """
        This function computes the accuracy using the classification accuracy.

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
        return np.mean(y_true == y_pred)
    
    def _get_accuracy_regression(self, y_true, y_pred):
        """
        This function computes the accuracy accuracy using the r2 score.

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
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        return r2
    
    def accuracy(self,y_true, y_pred):
        """
        This function computes the accuracy of the model.

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
        if self.type == 'classification':
            return self._get_accuracy_classification(y_true, y_pred)
        elif self.type == 'regression':
            return self._get_accuracy_regression(y_true, y_pred)
        else:
            raise ValueError('type must be either classification or regression')
        
    def score(self, x, y):
        """
        This function computes the accuracy of the model.

        Parameters:
        -----------
        x: ArrayLike
            The input data of shape (m,n) where m is the number of samples and n is the number of features.
            
        y: ArrayLike
            The input data of shape (m,1) where m is the number of samples.

        Returns:
        --------
        The accuracy of the model.
        """
        y_pred = self.predict(x)
        return self.accuracy(y, y_pred)
    
    def __repr__(self) -> str:
        return f'KNN(k={self.k}, type={self.type})'
    
    def __str__(self) -> str:
        return self.__repr__()