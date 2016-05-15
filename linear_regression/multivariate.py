
# Thomio Watanabe
# Date: May 2016
# Exercice from:
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


print '-- Loading dat aset files... '
x = np.loadtxt('dataset/ex3x.dat')
y = np.loadtxt('dataset/ex3y.dat')


print '-- Normalizing the input data'
mean = np.mean(x, axis = 0)
sigma = np.std(x, axis = 0)
x[:,0] = ( x[:,0] - mean[0] ) / sigma[0]
x[:,1] = ( x[:,1] - mean[1] ) / sigma[1]


m = y.size
print '-- Number of elements: ' + str(m)
x = np.array([ (np.ones(m)), x[:,0], x[:,1] ])


# Define the cost function
# Each set of coefficients(a) have a different cost
def cost_func(a = np.array([0,0,0]) ):
    # Hipothesis: h(x) = a0 + a1*x1 + a2*x2
    h = a.dot(x)
    # Cost: (1/2m) * sum( (h-y)^2 )
    cost = (1./(2.*m)) * (h - y).transpose().dot(h - y);
    return cost;


def gradient_descent(a = np.array([0,0,0]),alpha = 0.07,iterations = 1500):
    cost = np.zeros(iterations)
    # Calculate the gradient descent
    # Python matrix multiplication is array.dot()
    for i in range(iterations):
        h = a.dot(x)
        a = a - (alpha/m) * (h - y).dot(x.transpose())
        cost[i] = cost_func(a)
    print '-- Alpha: ' + str(alpha) + ' - Iterations: ' + str(iterations)
    print '-- Coefficients: ' + str(a)
    # Return the cost for each step (alpha)
    return a, cost;


print '-- -- --'
gd_args = (np.array([0,0,0]), 0.01, 50 )
# J = gradient_descent(a = np.array([0,0,0]),alpha = 0.01, iterations = 50)
a, J = gradient_descent( gd_args[0], gd_args[1], gd_args[2] )
print '-- Ploting the cost vs iterarions for alpha = ' + str(gd_args[1])
plt.figure(1), plt.subplot(411), plt.plot(range(50),J)
print '-- -- --'

gd_args = (np.array([0,0,0]), 0.03, 50 )
a, J = gradient_descent( gd_args[0], gd_args[1], gd_args[2] )
print '-- Ploting the cost vs iterarions for alpha = ' + str(gd_args[1])
plt.subplot(412), plt.plot(range(50),J)
print '-- -- --'

gd_args = (np.array([0,0,0]), 0.1, 50 )
a, J = gradient_descent( gd_args[0], gd_args[1], gd_args[2] )
print '-- Ploting the cost vs iterarions for alpha = ' + str(gd_args[1])
plt.subplot(413), plt.plot(range(50),J)
print '-- -- --'

gd_args = (np.array([0,0,0]), 0.3, 50 )
a, J = gradient_descent( gd_args[0], gd_args[1], gd_args[2] )
print '-- Ploting the cost vs iterarions for alpha = ' + str(gd_args[1])
plt.subplot(414), plt.plot(range(50),J), plt.show()
print '-- -- --'


# Best alpha value = 0.1
gd_args = (np.array([0,0,0]), 0.1, 1500 )
a, J = gradient_descent( gd_args[0], gd_args[1], gd_args[2] )
print '-- Final coefficients: ' + str(a)
h = a[0] + a[1]*(1650 - mean[0])/sigma[0] + a[2]*(3 - mean[1])/sigma[1]
print '-- Price house for 1650 ft and 3 bedrooms: ' + str(h)


# # Plot results
# x0 = x.transpose()[:,1]
# plt.plot(x0, y, 'o', x0, a.dot(x)), plt.show()
