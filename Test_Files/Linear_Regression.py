import matplotlib.pyplot as plt
import math
import time
from multiprocessing import Pool


def get_cost(t0, t1, x, i, y):
    return ((t0 + t1 * x[i - 1]) - y[i - 1]) ** 2


def cost_function(m, theta0, theta1, x, y):
    pool = Pool(processes=4)
    # Cong thuc trong bai giang:
    return sum([pool.apply(get_cost, (theta0, theta1, x, i, y)) for i in range(1, m + 1)]) / (2 * m)


# def cost_function(m, theta0, theta1, x, y):
# Cong thuc trong bai giang:
# return sum([((theta0 + theta1*x[i - 1]) - y[i - 1])**2 for i in range(1, m+1)]) / (2*m)


def update_equation(m, theta0, theta1, x, y):
    cost = cost_function(m, theta0[0], theta1[0], x, y)

    temp = [0, 0]
    # alpha = 0.005:
    temp[0] = 0.005 * cost
    temp[1] = 0.005 * cost

    theta0[0] -= temp[0]
    theta1[0] -= temp[1]


def get_distance_list(distance, i, x, y):
    for j in xrange(0, len(data)):
        distance.append(math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))
    distance.sort()

    value = 0
    for k in xrange(len(distance) / 10):
        value += distance[k]
    return [i, value]


def scale_data(data):
    x = []
    y = []

    for i in xrange(len(data)):
        x.append(float(data[i].split(',')[0]))
        y.append(float(data[i].split(',')[1]))

    pool = Pool(processes=4)
    result = [pool.apply(get_distance_list, ([], i, x, y)) for i in xrange(len(data) - 1)]
    distance_list = result

    distance_list.sort(key=lambda x: x[1], reverse=True)
    remove_list = [item for item in [distance_list[i][0] for i in range(0, len(distance_list) / 10)]]

    x1 = []
    y1 = []

    for i in range(0, len(x)):
        if i not in remove_list:
            x1.append(x[i])
            y1.append(y[i])

    return [x1, y1]
    # return [x[:end], y[:end]]


if __name__ == '__main__':
    start = time.time()

    data = open("ex1data1.txt", "r").readlines()

    x = []
    y = []

    for i in xrange(len(data)):
        x.append(float(data[i].split(',')[0]))
        y.append(float(data[i].split(',')[1]))

    # Load data:
    # for i in range(0, len(data)):
    # x.append(float(data[i].split(',')[0]))
    # y.append(float(data[i].split(',')[1]))

    m = len(x)  # So luong iterators
    # theta0 va theta1 la array cho de update(tham chieu):
    t0 = [0]  # theta0
    t1 = [1]  # theta1

    # t2 va t3 luu ket qua theta0 & theta1 cuoi cung tim duoc:
    t2 = 0
    t3 = 1

    min_cost = cost_function(m, t0[0], t1[0], x, y)  # Cost nho nhat
    current_cost = 0  # Cost tinh duoc sau moi lan chay

    # Bat dau chay:
    while True:
        update_equation(m, t0, t1, x, y)
        current_cost = cost_function(m, t0[0], t1[0], x, y)  # Tinh cost
        # Neu cost la nho nhat thi update min_cost, t2 va t3:
        if current_cost < min_cost:
            # Truong hop min_cost khong thay doi qua nhieu thi chap nhan ket qua chay:
            if min_cost - current_cost < 0.05:
                break
            # Update:
            min_cost = current_cost
            t2 = t0[0]
            t3 = t1[0]

    print 'Total time: ', time.time() - start
    # Ve:
    plt.plot(x, y, 'ro')
    plt.plot([4, 24], [t2 + 4 * t3, t2 + 24 * t3])
    plt.axis([4, 24, -5, 25])

    plt.show()
