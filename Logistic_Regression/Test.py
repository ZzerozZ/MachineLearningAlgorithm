from pymatrix import Matrix
import pandas as pd
import numpy as np
from Regularization import Hx, get_matrix

if __name__ == '__main__':
    df1 = pd.read_csv("result.txt").values
    df2 = (pd.read_csv("ex2data2.txt")).values

    temp = get_matrix("ex2data2.txt")
    X = temp[0]
    Y = temp[1]
    theta = Matrix.from_list(df1)
    for i in range(len(df1)):
        np.append(df1[i], 1-Hx(theta, X, 6))
    print df1