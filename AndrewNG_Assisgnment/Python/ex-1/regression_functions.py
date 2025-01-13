import numpy as np


def feature_normalize(X):
    """
    FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """

    x_norm = X
    mu = np.zeros([1, np.size(X, 1)], int)
    sigma = np.zeros([1, np.size(X, 1)], int)

    for i in range(np.size(X, 1)):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])

    for i in range(np.size(X, 1)):
        for j in range(np.size(X, 0)):
            x_norm[j, i] = x_norm[j, i] - mu[i]
            x_norm[j, i] = x_norm[j, i] / sigma[i]

    return x_norm, mu, sigma


def compute_cost(X, y, theta):
    """
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    error = X * theta - y  # calculate error
    Sq_error = np.square(error)  # calculate sq error

    #  Instructions: Compute the cost of a particular choice of theta
    #  You should set J to the cost.
    J = J + (sum(Sq_error)) / (2 * m)

    return J


def compute_cost_multi(X, y, theta):
    """
    COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    h = 0
    p = np.zeros(m, int)

    for i in range(m):
        for j in range(len(theta)):
            p[i] = p[i] + theta[j] * X[i, j]

    for i in range(m):
        h = h + (p[i] - y[i]) ** 2
        J = 0.5 * h / m

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    J_history = np.zeros([num_iters], int)

    for i in range(num_iters):
        # Instructions: Perform a single gradient step on the parameter vector theta
        error = X * theta - y  # calculate
        temp0 = theta[0] - sum(alpha * error / m)
        temp1 = theta[1] - sum(alpha * error.transpose() * X[:, 1] / m)
        theta = [temp0, temp1]

        # Save the cost J in every iteration
        J_history[iter] = compute_cost(X, y, theta)

    return theta, J_history


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = len(y)    # number of training examples
    J_history = np.zeros([num_iters, 1], int)

    for iter in range(num_iters):
        g = np.zeros(m, int)
        p = np.zeros(len(theta), int)

        for i in range(m):
            for j in range(len(theta)):
                g[i] = g[i] + theta[j] * X[i, j]

        for j in range(len(theta)):
            for i in range(m):
                p[i] = p[i] + (g[i] - y[i]) * X[i, j]
            theta[j] = theta[j] - alpha / m * p[j]

        # Save the cost J in every iteration
        J_history[iter] = compute_cost_multi(X, y, theta)

    return theta, J_history

