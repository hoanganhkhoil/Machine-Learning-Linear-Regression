# Author: Khoi Hoang
# Linear Regression-Not-Using-SKlearn

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# Create some sample data
def create_data(size, variance, step=2):
    val = 1
    ys = []
    for i in range(size):
        y = val + random.randrange(-variance,variance)
        val += step
        ys.append(y)

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# Find slope (m) and intercept (b)
def best_fit_slope_and_intercept(xs,ys):
    m = (mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs)**2 - mean(xs**2))
    b = mean(ys) - m * mean(xs)
    return m,b

# Find square error
def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original)**2)


def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original] # Create a vector whose each element is the mean of ys
    squared_error_prediction_and_ys = squared_error(ys_original, ys_line)
    squared_error_ys_and_y_mean = squared_error(ys_original, y_mean_line)
    return 1 - (squared_error_prediction_and_ys / squared_error_ys_and_y_mean)


xs , ys = create_data(50,10)


m,b = best_fit_slope_and_intercept(xs,ys)

prediction_line = (m*xs)+b  # same size as ys

r_squared = coefficient_of_determination(ys, prediction_line)

print (r_squared)

# Visualize the graph
plt.scatter(xs,ys)
plt.plot(xs,prediction_line)
plt.show()

