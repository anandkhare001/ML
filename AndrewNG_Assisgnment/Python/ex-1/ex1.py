import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from regression_functions import *

"""
Machine Learning Online Class - Exercise 1: Linear Regression

 Instructions
 ------------

 This file contains code that helps you get started on the
 linear exercise. You will need to complete the following functions
 in this exericse:

    warmUpExercise.m
    plotData.m
    gradientDescent.m
    computeCost.m
    gradientDescentMulti.m
    computeCostMulti.m
    featureNormalize.m
    normalEqn.m

 For this exercise, you will not need to change any code in this file,
 or any other files other than those mentioned above.

x refers to the population size in 10,000s
y refers to the profit in $10,000s

"""

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... ')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.loadtxt("ex1data1.txt", delimiter=',', dtype=float)
X = data[:, 0]
y = data[:, 1]
m = len(y)  # number of training examples
plt.plot(X, y)

# =================== Part 3: Cost and Gradient descent ===================
X = np.array([np.ones([m], int), data[:, 1]])  # Add a column of ones to x
theta = np.zeros([2], int)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...')

# compute and display initial cost
J = compute_cost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {}'.format(J))
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = compute_cost(X, y, [-1, 2])
print('With theta = [-1 ; 2]\nCost computed = {}'.format(J))
print('Expected cost value (approx) 54.24\n')

# run gradient descent
print('Running Gradient Descent ...\n')
theta = gradient_descent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: {}'.format(theta))
print('Expected theta values (approx)')
print(' -3.6303\n  1.1664\n')

# Plot the linear fit
plt.plot(X[:, 2], X * theta, '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta
print('For population = 35,000, we predict a profit of {}\n'.format(predict1 * 10000))
predict2 = [1, 7] * theta
print('For population = 70,000, we predict a profit of {}}\n'.format(predict2 * 10000))
print('Program paused. Press enter to continue.\n')

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.arange(-10, 10, 100)
theta1_vals = np.arange(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)], int)

# Fill out J_vals
for i in range(theta0_vals):
    for j in range(theta1_vals):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i, j] = compute_cost(X, y, t)

# Because of the way mesh grids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.transpose()
# Surface plot
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
plt.xlabel('\theta_0')
plt.ylabel('\theta_1')

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contourf(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('\theta_0')
plt.ylabel('\theta_1')
plt.plot(theta[0], theta[0], 'rx', 'MarkerSize', 10, 'LineWidth', 2)
plt.show()
