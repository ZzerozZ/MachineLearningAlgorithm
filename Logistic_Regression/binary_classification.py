# Demo for Binary Lienear Regression algorithm
# by Nghia Duong
# facebook.com/zzerodev

# This code demo for dataset with 2 feature and alpha = 0.0001


import pandas as pd
import numpy as np
from pymatrix import Matrix
import matplotlib.pyplot as plt


def init_theta(X, Y):
    """
    Initialization theta vector
    :param X:
    :param Y:
    :return: Theta vector + Matrix for any class
    """
    # Split X:
    X1 = Matrix.from_list([X[i] for i in range(X.numrows) if Y[i][0] == 1])
    X2 = Matrix.from_list([X[i] for i in range(X.numrows) if Y[i][0] == 0])

    # A and B are avg data point of X1 and X2:
    A = [sum([X1[i][1] for i in range(X1.numrows)])/X1.numrows, sum([X1[i][2] for i in range(X1.numrows)])/X1.numrows]
    B = [sum([X2[i][1] for i in range(X2.numrows)])/X2.numrows, sum([X2[i][2] for i in range(X2.numrows)])/X2.numrows]

    # Vector AB:
    AB = [B[0] - A[0], B[1] - A[1]]
    # I is midpoint between A and B:
    I = [(A[0] + B[0])/2, (A[1] + B[1])/2]

    # Get equation of AB's bisection:
    theta1 = np.array([[-(AB[0]*I[0] + AB[1]*I[1])], [AB[0]], [AB[1]]])

    # Scale theta:
    div = theta1[0][0]
    for i in range(len(theta1)):
        if abs(theta1[i][0]) > abs(div):
            div = theta1[i][0]

    for i in range(len(theta1)):
        theta1[i][0] /= div

    # Return theta and matrix of any class
    return [Matrix.from_list(theta1), X1, X2]


def get_matrix(data_input):
    """
    Init X and Y matrix
    :param data_input: input file path
    :return: Matrix X and Y
    """
    # Read data to dataframe df:
    df = (pd.read_csv(data_input)).values
    # Get X and Y:
    X = Matrix.from_list([df[i][0:2] for i in range(len(df))])
    Y = Matrix.from_list([df[i][2:] for i in range(len(df))])

    # Append 1 into any row of X:
    X = Matrix.from_list([np.insert(X[i], 0, (1,)) for i in range(len(df))])
    # Return:
    return [X, Y]


def sigmoid_function(theta, X):
    """
    Cacl sigmoid value
    :param theta:
    :param X:
    :return:
    """
    z = Matrix.from_list(X) * (-theta)
    return 1.0 / (1 + np.exp(z[0][0]))


def cost_function(X, Y, theta):
    """
    Cacl avg cost value
    :param X:
    :param Y:
    :param theta:
    :return: Cost value
    """
    m = X.numrows  # Data length
    # Cacl total cost:
    cost = ([(-Y[i][0] * np.log(sigmoid_function(theta, [X[i]]))
              - (1 - Y[i][0]) * np.log(1 - sigmoid_function(theta, [X[i]])))
             * Matrix.from_list([[X[i][0]]]) for i in range(m)])

    # Return avg cost:
    return -sum([cost[i][0][0] for i in range(len(cost))])/m


def update_equation(X, Y, theta, alpha):
    """
    Update Decision Boundary
    :param X:
    :param Y:
    :param theta:
    :param alpha:
    :return: New theta vector
    """
    return theta - Matrix.from_list([[alpha * cost_function(X, Y, theta)] for i in range(theta.numrows)])


if __name__ == '__main__':
    # Get necessary matrix(X, Y and theta):
    matrix = get_matrix("ex2data1.txt")
    X = matrix[0]
    Y = matrix[1]
    init_val = init_theta(X, Y)
    theta = init_val[0]
    # Set alpha value(for equation update later):
    alpha = 0.0001

    min_cost = cost_function(X, Y, theta)
    current_cost = 0

    # Learning:
    while True:
        # Update theta:
        theta = update_equation(X, Y, theta, alpha)
        # Cacl new cost:
        current_cost = cost_function(X, Y, theta)

        # If current theta is positive, update new cost:
        if abs(current_cost) < abs(min_cost):
            # :
            if abs(min_cost - current_cost) < 0.001 or abs(min_cost) < 0.001:
                break
            min_cost = current_cost

    X1 = init_val[1]
    X2 = init_val[2]

    # Draw result:
    plt.scatter(X1.trans()[1], X1.trans()[2], edgecolors='green')
    plt.scatter(X2.trans()[1], X2.trans()[2], edgecolors='red')

    plt.plot([20, 110], [-(theta[0][0] + theta[2][0]*20)/theta[1][0], -(theta[0][0] + theta[2][0]*110)/theta[1][0]])
    plt.axis([20, 110, 30, 110])

    plt.show()
