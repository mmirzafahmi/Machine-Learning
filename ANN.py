import matplotlib.pyplot as plt
import numpy as np
import os

data = [[3,  1.5, 1],
        [2,    1, 0],
        [4,  1.5, 1],
        [3,    1, 0],
        [3.5, .5, 1],
        [2,   .5, 0],
        [5.5,  1, 1],
        [1,    1, 0]]

mystery_data = [4.5, 1]

# Network

#           o    flower type
#     w1   / \   w2,b
# length  o   o  width

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def whichFlower(length, width):
    z = length * w1 + width * w2 + b
    pred = sigmoid(z)
    cost = np.square(pred - 1)
    if cost < .5:
        print("blue")
    else:
        print("red")

T = np.linspace(-6,6,100)

"""
plt.plot(T, sigmoid(T), c='r')
plt.plot(T, sigmoid_p(T), c='b')
"""

# training loop

learning_rate = 0.01

for i in range(100):
    rand_idx = np.random.randint(len(data))
    point = data[rand_idx]
    z = point[0] * w1 + point[1] * w2 + b

    pred = sigmoid(z)

    target = point[2]
    cost = np.square(pred - target)

    dcost_pred = 2 * (pred - target)
    dpred_dz = sigmoid_p(z)
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dz = dcost_pred * dcost_pred

    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

    if (cost <= 0.5):
        print(str(point) + ": Blue")
    else:
        print(str(point) + ": Red")
    print("\n")