import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.train_X.shape[0]):
                try:
                    distances[i, j] = abs(X[i] - self.train_X[j]).sum()
                except:
                    print(i, j)
        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances = np.zeros((X.shape[0], self.train_X.shape[0]))

        for i in range(X.shape[0]):
            distances[i] = abs(self.train_X - X[i]).sum(axis=1)
        
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances = abs(X[:, None, :] - self.train_X[None, :, :]).sum(axis=2)

        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        """

        pred = []
        nearest = []
        for i in range(distances.shape[0]):
            nearest_idx = np.argpartition(distances[i], self.k)[:self.k] # get indices if nearest objects in training set
            nearest_y = self.train_y[nearest_idx] # get classes of nearest objects
            pred.append((nearest_y == 0).sum() < (nearest_y == 1).sum()) # assign binaty class to analyzed object

        return np.array(pred)


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index
           for every test sample
        """
        pred = np.zeros(distances.shape[0], np.int64)

        for i in range(distances.shape[0]):
            nearest_idx = np.argpartition(distances[i], self.k)[:self.k] # get indices if nearest objects in training set
            nearest_y = self.train_y[nearest_idx] # get classes of nearest objects
            pred[i] = np.bincount(nearest_y).argmax() # assign binaty class to analyzed object
        
        return pred
