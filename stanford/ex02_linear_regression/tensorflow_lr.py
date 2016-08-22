
# Thomio Watanabe
# Date: May 2016
# Exercice from:
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Linear regression implementation in Google tensorflow
iterations = 1500
learning_rate = 0.07

train_x = np.loadtxt('../dataset/ex2x.dat')
train_y = np.loadtxt('../dataset/ex2y.dat')
n_elements = train_y.size

# Add column with 1's -> represent the a0 coefficient
train_x = np.array([ np.ones(n_elements), train_x])


X = tf.placeholder(tf.float32,shape=(2,n_elements))
Y = tf.placeholder(tf.float32)

# Hipothesis: h = a0 + a1*x
# coeffs = [a0, a1]
coeffs = tf.Variable( [[0., 0.]],dtype = tf.float32 )
hipothesis = tf.matmul(coeffs, X)
cost = tf.reduce_sum( tf.pow(hipothesis - Y, 2)) / (2. * n_elements)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# matrix = tf.constant([[1.],[2.]])
# mul = tf.matmul(coeffs, matrix)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(iterations):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})

    print sess.run(coeffs) 
    plt.plot(train_x[1,:], train_y,'o')
    fit = sess.run( tf.matmul(coeffs, tf.cast(train_x, tf.float32) ) )
    plt.plot(train_x[1,:], fit[0]), plt.show()
