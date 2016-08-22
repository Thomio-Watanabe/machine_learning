# Thomio Watanabe
# August, 2016
# Chapter 3 example

import tensorflow as tf
import numpy as np
import pickle


# The file has 3 groups of images
# 50000 images to train
# 10000 validation images
# 10000 test images
train_set, validation_set, test_set = pickle.load( open("mnist.pkl", "rb") )
print "Number of train images:", len(train_set[0])
print "Number of validation images:", len(validation_set[0])
print "Number of test images:", len(test_set[0])

# Each one of these groups have the ground truth
train_set_x, train_set_y = train_set
validation_set_x, validation_set_y = validation_set
test_set_x, test_set_y = test_set

# The placeholders will receive the data from the images variables during execution time
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

batch_size = 500
