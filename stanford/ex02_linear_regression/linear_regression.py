
# Thomio Watanabe
# Date: May 2016
# Exercice from:
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print '-- Loading dataset files... ' 
x = np.loadtxt('../dataset/ex2x.dat')
y = np.loadtxt('../dataset/ex2y.dat')

m = y.size
print '-- Number of elements: ' + str(m)

a = np.array([0,0])
x = np.array([ (np.ones(m)), (x) ])

# Partial derivative constant (alpha)
c = 0.07

# Max number of iterations
limit = 1500

# Calculate the gradient descent
# Python matrix multiplication is array.dot()
for i in range(limit):
    h = a.dot(x)
    a = a - (c/m) * (h - y).dot(x.transpose()) 

print '-- Linear regression coefficients: ' + str(a)

# Plot results
x0 = x.transpose()[:,1]
plt.plot(x0, y, 'o', x0, a.dot(x)), plt.show()


# Mean squared error J(a)
# J = 1/(2m) sum(h(a) - y)^2
a0 = np.arange(-3, 3, 0.05)
a1 = np.arange(-3, 3, 0.05)
J = np.ndarray(shape=(a0.size, a1.size), dtype=float)
for i in range(len(a0)):
    for j in range(len(a1)):
        summatory = 0
        for k in range(m):
            summatory = summatory + (1.0/(2.0 * m)) * np.power( a0[i] + a1[j] * x[1][k] - y[k], 2)
        J[i,j] = summatory
# Plot the mean squared error
A0, A1 = np.meshgrid(a0, a1)
ax = Axes3D(plt.gcf())
ax.plot_surface(A0, A1,J)
plt.show()
