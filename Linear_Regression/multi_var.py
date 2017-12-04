from pymatrix import Matrix
from time import time


def load_data(data, argc):
    """
    Lay ma tran X, Y
    :param data: list record doc tu file
    :param argc: so luong tham so moi record
    :return: ma tran chua tat ca tham so dau vao
    """
    temp = []
    for k in range(argc):
        temp.append(([float(data[i].split(',')[k]) for i in range(len(data))]))
    return temp


def get_matrix(var):
    """
    Lay ma tran theta, X va Y
    :param var: Ma tran tham so dau vao
    :return: list 3 ma tran: theta, X va Y
    """
    # Y la cot cuoi cung cua var, con lai thuoc X
    X = [var[i] for i in range(len(var) - 1)]
    Y = var[len(var) - 1]

    # Them value 1 vao dau moi dong cua X
    X.insert(0, [1 for i in range(len(var[0]))])
    # Tao ma tran X va Y:
    mX = Matrix.from_list(X)
    mY = Matrix.from_list([Y])

    # Tra ve theta, X va Y:
    return [((mX * mX.trans()).inv()) * mX * mY.trans(), mX, mY]


def cost_function(theta, X, Y):
    """
    Tinh cost cua ham so dang xet
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    # Tinh cost tung phan tu:
    total = (X * theta) - Y
    # Binh phuong total[i]:
    temp = sum(total[i][0]**2 for i in range(total.numrows))

    # Tra ve cost cuoi cung
    return temp/(2*X.numrows)


def update_equation(theta, X, Y, alpha):
    """
    Update theta
    :param theta:
    :param X:
    :param Y:
    :param alpha:
    :return: new theta
    """
    # theta -= (alpha / m)(X.trans() * (X*theta - Y))
    return theta - ((alpha/X.numrows) * (X.trans() * (X * theta - Y)))


if __name__ == '__main__':
    # Bat dau tinh gio
    start = time()
    # Doc data tu file:
    data = open("ex1data2.txt", "r").readlines()
    # Load data vao ma tran:
    len1 = len(data[0].split(','))
    var = load_data(data, len1)
    # Tao ma tran theta, X va Y
    matrix = get_matrix(var)
    theta = matrix[0]
    X = matrix[1].trans()
    Y = matrix[2].trans()

    print 'Pre-processing time: ', time() - start

    min_cost = cost_function(theta, X, Y)  # Cost nho nhat
    current_cost = 0  # Cost tinh duoc sau moi lan chay

    # Bat dau chay:
    while True:
        theta = update_equation(theta, X, Y, 0.00001)
        current_cost = cost_function(theta, X, Y)
        # Neu cost la nho nhat thi update min_cost, t2 va t3:
        if current_cost < min_cost:
            # Truong hop min_cost khong thay doi qua nhieu thi chap nhan ket qua chay:
            if min_cost - current_cost < 450:
                break
            # Update:
            min_cost = current_cost

    print 'Total time: ', time() - start
    open("result.txt", "w").write(str(theta.trans()))

