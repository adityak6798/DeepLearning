# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:53:40 2018

@author: Aditya Khandelwal
"""

#Imports
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
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

#The Data
input_x = np.random.rand(n_examples, n_input)
input_y = np.random.rand(n_examples, n_output)

#Weights and biases as Theano Shared Variables, so that they can be updated
W1 = theano.shared(np.random.standard_normal((n_input, n_hidden)),name = 'W1')
W2 = theano.shared(np.random.standard_normal((n_hidden, n_output)),name='W2')
b1 = theano.shared(np.zeros(n_hidden),name='b1')
b2 = theano.shared(np.zeros(n_output),name='b2')

#Placeholder like matrices that will be given to the graph at runtime
X = T.dmatrix('X')
Y = T.dmatrix('Y')

#The Graph
Z1 = T.dot(X,W1) + b1
Z2 = T.dot(Z1,W2) + b2
Yhat = 1 / (1 + T.exp(-Z2))
loss = T.sum(T.sqr(Y-Yhat))

#Weights and biases updates, by using theano to automatically compute gradients and apply them as per your own optimizer equation 
update_W1 = W1 - learning_rate*T.grad(loss,W1)
update_W2 = W2 - learning_rate*T.grad(loss,W2)
update_b1 = b1 - learning_rate*T.grad(loss,b1)
update_b2 = b2 - learning_rate*T.grad(loss,b2)

#Define a function in theano to compute the graph given certain inputs, update certain nodes, and output certain values
train = theano.function(inputs=[X,Y],updates=[(b1,update_b1),(b2,update_b2),(W1,update_W1),(W2,update_W2)],outputs=[loss])

err = []

#The training procedure, by feeding batches to the training function
for epoch in range(num_epochs):
    for iteration in range(num_iterations):
        X_batch = input_x[iteration*batch_size:iteration*batch_size+batch_size,:]
        Y_batch = input_y[iteration*batch_size:iteration*batch_size+batch_size,:]
        _ = train(X_batch,Y_batch)
    if epoch % 10 == 0:
        err.append(train(input_x,input_y)[0]/n_examples)

#Plot the error
plt.plot(err)
