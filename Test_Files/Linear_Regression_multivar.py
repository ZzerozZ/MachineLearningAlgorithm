from pymatrix import Matrix
from time import time
from multiprocessing import Process


def cost_function(m, theta0, theta1, theta2, x1, x2, y):
    cost = 0
    for i in range(1, m+1):
        # Cong thuc trong bai giang:
        cost += 1.0*((theta0[0] + theta1[0]*x1[i - 1] + theta2[0]*x2[i - 1]) - y[i - 1])**2

    return 1.0*cost/(2*m)


def update_equation(m, theta0, theta1, theta2, x1, x2, y, alpha):
    # Update theta0:
    temp = [0, 0, 0]
    for i in range(1, m+1):
        temp[0] += 1.0*((theta0[0] + theta1[0]*x1[i - 1] + theta2[0]*x2[i - 1]) - y[i - 1])

    # Update theta01:
    for i in range(1, m + 1):
        temp[1] += 1.0/m * ((theta0[0] + theta1[0] * x1[i - 1] + theta2[0] * x2[i - 1]) - y[i - 1]) * x1[i - 1]

    # Update theta2:
    for i in range(1, m + 1):
        temp[2] += 1.0 * ((theta0[0] + theta1[0] * x1[i - 1] + theta2[0] * x2[i - 1]) - y[i - 1]) * x2[i - 1]

    theta0[0] -= (alpha / m) * temp[0]
    theta1[0] -= (alpha / m) * temp[1]
    theta2[0] -= (alpha / m) * temp[2]


# Init theta value:
def get_theta(m, x1, x2, y):
    t = Matrix(m, 3, fill=1)
    for i in range(0, m):
        t[i][1] = x1[i]
        t[i][2] = x2[i]

    k = Matrix(m, 1, fill=0)
    for i in range(0, m):
        k[i][0] = y[i]

    return ((t.trans() * t).inv()) * t.trans() * k


if __name__ == '__main__':
    start = time()

    data = open("ex1data2.txt", "r").readlines()

    m = len(data)  # So luong iterators

    x1 = []
    x2 = []
    y = []
    # Load data:
    for i in range(0, len(data)):
        x1.append(float(data[i].split(',')[0]))
        x2.append(float(data[i].split(',')[1]))
        y.append(float(data[i].split(',')[2]))

    theta = get_theta(m, x1, x2, y)

    t0 = theta[0]
    t1 = theta[1]
    t2 = theta[2]

    r1 = 0
    r2 = 0
    r3 = 0

    for i in range(0, len(x1)):
        x1[i] = x1[i]

    min_cost = cost_function(m, t0, t1, t2, x1, x2, y)  # Cost nho nhat
    current_cost = 0  # Cost tinh duoc sau moi lan chay

    # Bat dau chay:
    while True:
        update_equation(m, t0, t1, t2, x1, x2, y, 0.00002)
        current_cost = cost_function(m, t0, t1, t2, x1, x2, y)
        # Neu cost la nho nhat thi update min_cost, t2 va t3:
        if current_cost < min_cost:
            # Truong hop min_cost khong thay doi qua nhieu thi chap nhan ket qua chay:
            if min_cost - current_cost < 450:
                break
            # Update:
            min_cost = current_cost
            r1 = t0[0]
            r2 = t1[0]
            r3 = t2[0]

    print r1 + r2*2609 + r3*4, ' ', time()-start
