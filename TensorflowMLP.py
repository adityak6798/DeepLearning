# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 22:26:03 2018

@author: Aditya Khandelwal
"""

#Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Variables
n_input = 5
n_examples = 2000
n_output = 1
n_hidden = 6
learning_rate = 0.001
batch_size = 10
num_epochs = 200
num_iterations = n_examples//batch_size
error = []

#The Data
input_x = np.random.rand(n_examples, n_input)
input_y = np.random.rand(n_examples, n_output)

#Placeholders
X = tf.placeholder(tf.float32,shape=(batch_size,n_input))
Y = tf.placeholder(tf.float32,shape=(batch_size,n_output))

#Weights and Biases
W1 = tf.Variable(tf.random_normal((n_input,n_hidden)))
b1 = tf.Variable(tf.ones(n_hidden))
W2 = tf.Variable(tf.random_normal((n_hidden,n_output)))
b2 = tf.Variable(tf.ones(n_output))

#the Computation Graph
Z1 = tf.matmul(X,W1) + b1
Z2 = tf.nn.sigmoid(tf.matmul(Z1,W2) + b2)

#The loss function
loss = tf.reduce_mean(tf.square(Y-Z2))

#The optimizer
optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optim.minimize(loss)

#Innitialising all variables, necessary to do before running the model
init = tf.global_variables_initializer()

#Running the Session
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        for iteration in range(num_iterations):
            X_batch = input_x[iteration*batch_size:iteration*batch_size+batch_size,:]
            Y_batch = input_y[iteration*batch_size:iteration*batch_size+batch_size,:]
            sess.run(train,feed_dict={X:X_batch,Y:Y_batch}) #Trainiing on current batch
        if epoch % 10 == 0:
            err = 0
            for iteration in range(num_iterations): #To calculate the error over all training examples
                X_batch = input_x[iteration*batch_size:iteration*batch_size+batch_size,:]
                Y_batch = input_y[iteration*batch_size:iteration*batch_size+batch_size,:]
                err += sess.run(loss,feed_dict={X:X_batch,Y:Y_batch})
            error.append(err)
            
#Visualisation            
plt.plot(error)