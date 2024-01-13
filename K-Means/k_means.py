import numpy as np

class KMeans:
    """
    K-Means clustering algorithm

    Parameters:
    ----------
    n_clusters : int, default=2
        Number of clusters
    
    tol : float, default=0.001
        Relative tolerance with regards to inertia to declare convergence

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run

    Attributes:
    ----------
    centroids : dict, shape = [n_clusters, n_features]
        Coordinates of cluster centers

    classifications : dict, shape = [n_clusters, n_features]
        Coordinates of cluster centers

    Methods:
    -------
    fit(x)
        Compute k-means clustering

    predict(x)
        Predict the closest cluster each sample in x belongs to

    accuracy(y_true, y_pred)
        Return the mean accuracy on the given test data and labels

    score(x, y)
        Return the mean accuracy on the given test data and labels
    """
    def __init__(self, n_clusters=2, tol=0.001, max_iter=300):
        """
        Initialize K-Means clustering algorithm

        Parameters:
        ----------
        n_clusters : int, default=2
            Number of clusters

        tol : float, default=0.001
            Relative tolerance with regards to inertia to declare convergence

        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a single run

        Returns
        -------
        None
        """
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, x):
        """
        Compute k-means clustering

        Parameters:
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training instances to cluster

        Returns
        -------
        None
        """
        self.centroids = {}
        self.classifications = {}
        rnd = np.random.default_rng().choice(len(x), self.n_clusters)

        for i, centroid in enumerate(rnd):
            self.centroids[i] = x[centroid]
            self.classifications[i] = []

        for i in range(self.max_iter):
            for classification in self.classifications:
                self.classifications[classification] = []

            for row in x:
                distances = [
                    np.linalg.norm(row - self.centroids[centroid])
                    for centroid in self.centroids
                ]
                classification = np.argmin(distances)
                self.classifications[classification].append(row)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0
                )

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum(
                    (current_centroid - original_centroid) / original_centroid * 100.0
                ) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, x):
        """
        Predict the closest cluster each sample in x belongs to

        Parameters:
        ----------
        x : array-like, shape = [n_samples, n_features]
            New data to predict

        Returns
        -------
        classification : array-like, shape = [n_samples]
            Index of the cluster each sample belongs to
        """
        classifications = []
        for row in x:
                distances = [
                    np.linalg.norm(row - self.centroids[centroid])
                    for centroid in self.centroids
                ]
                classification = np.argmin(distances)
                classifications.append(classification)
        return classifications
    
    def accuracy(self, y_true, y_pred):
        """
        Return the mean accuracy on the given test data and labels

        Parameters:
        ----------
        y_true : array-like, shape = [n_samples]
            True labels for x

        y_pred : array-like, shape = [n_samples]
            Predicted labels for x

        Returns
        -------
        accuracy : float
            Mean accuracy of self.predict(x) wrt. y.
        """
        return np.mean(y_true == y_pred)
    
    def score(self, x, y):
        """
        This function computes the accuracy of the model.

        Parameters::
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
    
    def __repr__(self):
        return f"KMeans(k={self.n_clusters}, tol={self.tol}, max_iter={self.max_iter})"
    
    def __str__(self):
        return self.__repr__()
