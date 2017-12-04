# Demo for Regularization for Logistic Regression algorithm
# by Nghia Duong
# facebook.com/zzerodev

# This code demo for dataset with 2 feature, degree = 6, alpha = 0.001 and lambda = 1


from pymatrix import Matrix
import pandas as pd
import numpy as np


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
    A = [sum([X1[i][1] for i in range(X1.numrows)]) / X1.numrows,
         sum([X1[i][2] for i in range(X1.numrows)]) / X1.numrows]
    B = [sum([X2[i][1] for i in range(X2.numrows)]) / X2.numrows,
         sum([X2[i][2] for i in range(X2.numrows)]) / X2.numrows]

    # Vector AB:
    AB = [B[0] - A[0], B[1] - A[1]]
    # I is midpoint between A and B:
    I = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2]

    # Get equation of AB's bisection:
    theta1 = [[-(AB[0] * I[0] + AB[1] * I[1])], [AB[0]], [AB[1]], [1], [0], [1]]
    [theta1.append([0]) for i in range(22)]  # )

    # Find max value of element in theta:
    max_value = theta1[0][0]
    for i in range(len(theta1)):
        if abs(theta1[i][0]) > abs(max_value):
            max_value = theta1[i][0]

    # Scale theta:
    for i in range(len(theta1)):
        theta1[i][0] /= max_value

    # Return theta and matrix of any class
    return [Matrix.from_list(theta1), X1, X2]


def Hx(theta, X, degree):
    """
    Get h(x) value from theta(1*m matrix) and X(m*1 matrix)
    :param theta:
    :param X:
    :param degree: max degree of equation
    :return:
    """
    total = 0
    k = 0  # Theta iterator

    # Cacl h(x) value:
    for i in range(degree + 1):
        for j in range(i + 1):
            total += (X[0][1] ** j) * (X[0][2] ** (i - j)) * theta[k][0]  # x1^m + x2^n with m+n = degree
            k += 1  # Increasing k

    # Return:
    return total


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
    z = Hx(-theta, X, 6)
    return 1.0 / (1 + np.exp(z))


def cost_function(X, Y, theta, _lambda):
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
              - (1 - Y[i][0]) * np.log(1 - sigmoid_function(theta, [X[i]]))) * Matrix.from_list([[X[i][0]]]) for i in
             range(m)])

    # Regularization value:
    regularization = (_lambda / (2 * m)) * sum([theta[i][0] ** 2 for i in range(theta.numrows)])

    # Return avg cost:
    return -sum([cost[i][0][0] for i in range(len(cost))]) / m + regularization


def update_equation(X, Y, theta, alpha, _lambda):
    """
    Update Decision Boundary
    :param X:
    :param Y:
    :param theta:
    :param alpha:
    :return: New theta vector
    """
    temp = theta[0][0]  # Save theta0 value

    # Update theta(included theta0):
    theta = theta - Matrix.from_list([[alpha * cost_function(X, Y, theta, _lambda)] for i in range(theta.numrows)])

    # Update theta0 by special way:
    theta[0][0] = temp - alpha * cost_function(X, Y, theta, 0)

    return theta


if __name__ == '__main__':
    # Get necessary matrix(X, Y and theta):
    matrix = get_matrix("ex2data2.txt")

    X = matrix[0]
    Y = matrix[1]

    init_val = init_theta(X, Y)
    theta = init_val[0]
    # Set alpha value(for equation update later):
    alpha = -0.001
    _lambda = 1

    min_cost = cost_function(X, Y, theta, _lambda)
    current_cost = 0

    # Learning:
    while True:
        # Update theta:
        theta = update_equation(X, Y, theta, alpha, _lambda)
        # Cacl new cost:
        current_cost = cost_function(X, Y, theta, _lambda)

        # If current theta is positive, update new cost:
        if abs(current_cost) < abs(min_cost):
            # :
            if abs(min_cost - current_cost) < 0.01 or abs(min_cost) < 0.001:
                break
            min_cost = current_cost

    # open("result.txt", "w").write(str(theta.trans()))

    df1 = pd.read_csv("ex2data2.txt").values
    for i in range(len(df1)):
        print [df1[i][j] for j in range(3)], '\t', abs(1 - Hx(theta, [X[i]], 6))