
# Thomio Watanabe
# Date: May 2016
# Exercice from:
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html

import numpy as np
import matplotlib.pyplot as plt


print '-- Loading dataset files... ' 
x = np.loadtxt('dataset/ex2x.dat')
y = np.loadtxt('dataset/ex2y.dat')

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
