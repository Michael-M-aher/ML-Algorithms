import numpy as np

class GBNaiveBayes:
    """
    Gaussian Naive Bayes classifier

    Parameters
    ----------
    None

    Attributes
    ----------
    class_labels : array-like, shape = [n_classes]
        Class labels

    class_priors : array-like, shape = [n_classes]
        Prior probabilities of class labels

    class_means : dict, shape = [n_classes, n_features]
        Means of each feature per class

    class_variances : dict, shape = [n_classes, n_features]
        Variances of each feature per class

    Methods
    -------
    fit(x, y)
        Fit the model according to the given training data

    predict(x)
        Perform classification on an array of test vectors x

    accuracy(y_true, y_pred)
        Return the mean accuracy on the given test data and labels
    
    score(x, y)
        Return the mean accuracy on the given test data and labels

    """

    def __init__(self):
        """
        Initialize Gaussian Naive Bayes classifier

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.class_labels = None
        self.class_priors = None
        self.class_means = None
        self.class_variances = None

    def fit(self, x, y):
        """
        Fit the model according to the given training data

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors

        y : array-like, shape = [n_samples]
            Target values

        Returns
        -------
        None
        """
        self.class_labels = np.unique(y)
        self.class_priors = np.array([np.mean(y == label) for label in self.class_labels])

        self.class_means = {}
        self.class_variances = {}

        for label in self.class_labels:
            class_data = x[y == label]
            self.class_means[label] = np.mean(class_data, axis=0)
            self.class_variances[label] = np.var(class_data, axis=0)

    def _calculate_likelihood(self, x, mean, variance):
        """
        Calculate likelihood of x given mean and variance

        Parameters
        ----------
        x : array-like, shape = [n_features]
            Training vector
        
        mean : array-like, shape = [n_features]
            Mean of each feature per class
        
        variance : array-like, shape = [n_features]
            Variance of each feature per class

        Returns
        -------
        likelihood : array-like, shape = [n_features]
            Likelihood of x given mean and variance
        """
        exponent = -((x - mean) ** 2) / (2 * variance)
        return np.exp(exponent) / np.sqrt(2 * np.pi * variance)

    def predict(self, x):
        """
        Perform classification on an array of test vectors x

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test vectors

        Returns
        -------
        predictions : array-like, shape = [n_samples]
            Predicted class label per sample
        """
        predictions = []
        for label in self.class_labels:
            prior = self.class_priors[label]
            likelihoods = np.prod(self._calculate_likelihood(x, self.class_means[label], self.class_variances[label]), axis=1)
            posterior = prior * likelihoods
            predictions.append(posterior)

        predictions = np.array(predictions).T
        return self.class_labels[np.argmax(predictions, axis=1)]
        
    def accuracy(self,y_true, y_pred):
       """
        Return the mean accuracy on the given test data and labels

        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            True labels for x

        y_pred : array-like, shape = [n_samples]
            Predicted labels for x

        Returns
        -------
        accuracy : float
            Mean accuracy of self.predict(x) wrt. y
        """
       return np.mean(y_true == y_pred)
        
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
        return "Gaussian Naive Bayes"
    
    def __str__(self) -> str:
        return self.__repr__()