import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    X_norm, mu, sigma = 0,0,0
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    mu_arr = np.ones((X.shape[0], X.shape[1]))*mu
    sigma_arr = np.ones((X.shape[0], X.shape[1]))*sigma
    X_norm = (X - mu_arr)/sigma_arr

# ============================================================

    return X_norm, mu, sigma
