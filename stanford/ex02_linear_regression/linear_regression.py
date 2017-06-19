
# Thomio Watanabe
# May 2016
# Exercice from:
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def iteration_implementation( x, y ):
    n = y.size

    # hypothesis => h = ax + b
    a = 0.0
    b = 0.0
    # Learning rate
    alpha = 0.01
    # Max number of iterations
    max_iter = 10000

    # Mean squared error (MSE)
    # 1/2n * sum(h - y)^2
    # derivative => 1/n * sum(h - y) * h'

    # Update function
    # a <- a - alpha * increment
    # b <- b - alpha * increment

    # increment = derivative * x

    for i in range(max_iter):
        acc_a = 0.0
        acc_b = 0.0
        for j in range( n ):
            h = a*x[j] + b
            acc_a += (h - y[j]) * x[j]
            acc_b += (h - y[j])
        a = a - alpha * (acc_a / n)
        b = b - alpha * (acc_b / n)

    # Plot results
    print '-- Linear regression coefficients: ', b, a
    plt.plot(x, y, 'o', x, a*x + b), plt.show()
    return


def matrix_implementation( x, y ):
    n = y.size

    # Matrix implementation
    a = np.array([0,0])
    x = np.array([ (np.ones(n)), (x) ])

    # Partial derivative constant (alpha)
    c = 0.07

    # Max number of iterations
    limit = 1500

    # Calculate the gradient descent
    # Python matrix multiplication is array.dot()
    for i in range(limit):
        h = a.dot(x)
        a = a - (c/n) * (h - y).dot(x.transpose())

    print '-- Linear regression coefficients: ' + str(a)

    # Plot results
    x0 = x.transpose()[:,1]
    plt.plot(x0, y, 'o', x0, a.dot(x)), plt.show()
    return


def plot_loss_surface( x, y ):
    n = y.size

    # Mean squared error J(a)
    # J = 1/(2n) sum(h(a) - y)^2
    a0 = np.arange(-3, 3, 0.05)
    a1 = np.arange(-3, 3, 0.05)
    J = np.ndarray(shape=(a0.size, a1.size), dtype=float)
    for i in range(len(a0)):
        for j in range(len(a1)):
            summatory = 0
            for k in range(n):
                summatory += (1.0/(2.0 * n)) * np.power( a0[i] + a1[j] * x[k] - y[k], 2)
            J[i,j] = summatory
    # Plot the mean squared error
    A0, A1 = np.meshgrid(a0, a1)
    ax = Axes3D(plt.gcf())
    ax.plot_surface(A0, A1,J)
    plt.show()
    return


if __name__ == "__main__":
    print '-- Loading dataset files... '
    x = np.loadtxt('../dataset/ex2x.dat')
    y = np.loadtxt('../dataset/ex2y.dat')

    n = y.size
    print '-- Number of samples: ' + str(n)

    iteration_implementation( x, y )
    matrix_implementation( x, y )
    plot_loss_surface( x, y )
