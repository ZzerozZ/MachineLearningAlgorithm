import matplotlib.pyplot as plt
import time


def scale_data(data):
    """
    Xoa cac diem gay nhieu
    :param data: list data doc tu file input
    :return: ma tran chua du lieu dau vao
    """
    # Lay ma tran cac tham so:
    var = ([float(data[i].split(',')[0]) for i in range(len(data))], [float(data[i].split(',')[1]) for i in range(len(data))])
    # Lay trung binh ti le x/y:
    avg = sum([var[0][i] / var[1][i] for i in range(len(data))]) / len(data)
    # Lay list input chuan:
    result = [[], []]
    for i in range(len(data)):
        # Neu ti le x/y khong qua khac ti le trung binh thi nhan diem i:
        if True:  # abs(avg - (var[0][i] / var[1][i])) < 4:
            result[0].append(var[0][i])
            result[1].append(var[1][i])

    return result


def cost_function(var, theta):
    """
    Tinh cost cua diem hien tai
    :param var: ma tran chua du lieu dau vao
    :param theta: vector chua theta0, theta1
    :return: avg(cost) / 2
    """
    sum = 0  # cost tong cong
    # Tinh tong cost:
    for i in range(len(var[0])):
        sum += ((theta[0] + theta[1] * var[0][i]) - var[1][i]) ** 2
    # return sum / 2*m: m la so luong record:
    return sum / (2 * len(var[0]))


def update_equation(var, theta):
    """
    Update theta
    :param var: ma tran chua du lieu dau vao
    :param theta: vector chua theta0, theta1
    """
    cost = cost_function(var, theta)  # Lay cost hien tai

    # Update theta:
    for i in range(len(theta)):
        theta[i] -= 0.0005 * cost


if __name__ == '__main__':
    # Doc data tu file:
    data = open("ex1data1.txt", "r").readlines()

    # Bat dau tinh gio:
    start = time.time()
    # Lay du lieu vao ma tran var:
    var = scale_data(data)

    print 'Scale time: ', time.time() - start

    theta = [0, 1]  # Theta tinh toan
    return_value = [0, 1]  # Gia tri theta cuoi cung

    min_cost = cost_function(var, theta)  # Cost nho nhat
    current_cost = 0  # Cost tinh duoc sau moi lan chay

    while True:
        update_equation(var, theta)
        current_cost = cost_function(var, theta)  # Tinh cost
        # Neu cost la nho nhat thi update min_cost, t2 va t3:
        if current_cost < min_cost:
            # Truong hop min_cost khong thay doi qua nhieu thi chap nhan ket qua chay:
            if min_cost - current_cost < 0.05:
                break
            # Update:
            min_cost = current_cost

    print 'Total time: ', time.time() - start
    # Ve:
    plt.plot(var[0], var[1], 'ro')
    plt.plot([4, 24], [theta[0] + 4 * theta[1], theta[0] + 24 * theta[1]])
    plt.axis([4, 24, -5, 25])

    plt.show()
