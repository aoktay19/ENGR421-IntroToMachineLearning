import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# read data into memory
from numpy import ndarray

data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter=",")
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter=",")

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 0.0
maximum_value = 2.0
x_interval = np.arange(start=minimum_value, stop=maximum_value + 0.002, step=0.002)


def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x_train, y_train, "b.", markersize=10)
    plt.plot(x_test, y_test, "r.", markersize=10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return (fig)


# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below

    y_hat = np.zeros(len(x_query), )
    for i in range(len(x_query)):
        for j in range(len(left_borders)):
            if left_borders[j] < x_query[i] <= right_borders[j]:
                y_hat[i] = np.mean(y_train[(x_train > left_borders[j]) & (x_train <= right_borders[j])])
                break

    # your implementation ends above
    return (y_hat)


bin_width = 0.10
left_borders = np.arange(start=minimum_value, stop=maximum_value, step=bin_width)
right_borders = np.arange(start=minimum_value + bin_width, stop=maximum_value + bin_width, step=bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches="tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat) ** 2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))


# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below

    y_hat = np.asarray([np.mean(y_train[(x_train > x_query[x] - bin_width*0.5) & (x_train <= x_query[x] + bin_width*0.5)]) for x in range(len(x_query))])

    # your implementation ends above
    return y_hat


bin_width = 0.10

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches="tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat) ** 2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))


# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below

    y_hat = np.asarray(
        [np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * ((x - x_train) / bin_width) ** 2) * y_train) /
         np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * ((x - x_train) / bin_width) ** 2)) for x in x_query])
    # your implementation ends above
    return (y_hat)


bin_width = 0.02

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches="tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat) ** 2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
