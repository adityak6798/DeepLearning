# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:12:38 2018

@author: Aditya Khandelwal

This file contains SGD, SGD with Momentum, AdaGrad, RMSProp, Adam implementations.
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt

#Parameters
n_input = 5
n_hidden = 2
n_output = 1
n_examples = 1000
num_epochs = 750
learning_rate = 0.5
batch_size = 50
num_iterations = 1000//batch_size
err = []
err_momentum = []

# Essentilly random data, can be replaced by true data
X = np.random.randn(n_input,n_examples)
Y = np.random.randn(n_output, n_examples)
#Y = 4*np.sum(X,axis=0) + np.random.randn(1,n_examples)

#Weights and biases initialisation
W1 = 2*np.random.rand(n_input,n_hidden)-1
b1 = 2*np.ones(shape=(n_hidden,1))-1
W2 = 2*np.random.rand(n_hidden,n_output)-1
b2 = 2*np.ones(shape=(n_output,1))-1

#Gradients initialisation
wg1 = np.ndarray(shape = W1.shape)
wg2 = np.ndarray(shape = W2.shape)
bg1 = np.ndarray(shape = b1.shape)
bg2 = np.ndarray(shape = b2.shape)

#Activation Functions
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

#Cost Function
def mse(pred_val,true_val):
    mat = (pred_val-true_val)/(2*pred_val.shape[1])
    #print(pred_val.shape[0])
    return np.sum(mat**2,axis=1)

#Training with MiniBatch Gradient Descent
for j in range(num_epochs):
    for i in range(num_iterations):
        #Grab a batch
        X_batch = X[:,i*batch_size:i*batch_size+batch_size]
        Y_batch = Y[:,i*batch_size:i*batch_size+batch_size]

        #Forward Propogation
        h1 = sigmoid(np.matmul(W1.transpose(),X_batch) + b1)
        op = np.matmul(W2.transpose(),h1) + b2

        #Error Calculation
        error = mse(op, Y_batch)

        #Gradients Computation
        op_grad = np.subtract(op, Y_batch)/batch_size

        W2_grad = np.matmul(h1,op_grad.transpose())

        b2_grad = np.sum(op_grad,axis=1).reshape(b2.shape)

        h1_grad = np.matmul(W2,op_grad)

        W1_grad = np.matmul(X_batch, (h1_grad*h1*(1-h1)).transpose())

        b1_grad = np.sum(h1_grad * h1 * (1-h1),axis=1).reshape(b1.shape)

        #Weights Updation
        W2 -= learning_rate*W2_grad

        W1 -= learning_rate*W1_grad

        b2 -= learning_rate*b2_grad

        b1 -= learning_rate*b1_grad

    #Report the training error every 50 epochs

    if j % 50 == 0:
        h1 = sigmoid(np.matmul(W1.transpose(),X) + b1)
        op = np.matmul(W2.transpose(),h1) + b2
        error = mse(op, Y)
        err.append(error)
        print("Epoch {}: Error : {}".format(j,error))

#Training with Gradient Descent With Momentum

#Weights and biases initialisation
W1 = 2*np.random.rand(n_input,n_hidden)-1
b1 = 2*np.ones(shape=(n_hidden,1))-1
W2 = 2*np.random.rand(n_hidden,n_output)-1
b2 = 2*np.ones(shape=(n_output,1))-1

#Gradients initialisation
wg1 = np.ndarray(shape = W1.shape)
wg2 = np.ndarray(shape = W2.shape)
bg1 = np.ndarray(shape = b1.shape)
bg2 = np.ndarray(shape = b2.shape)

W2_prev_grad = np.zeros(W2.shape)
b2_prev_grad = np.zeros(b2.shape)
W1_prev_grad = np.zeros(W1.shape)
b1_prev_grad = np.zeros(b1.shape)
momentum = 0.9

for j in range(num_epochs):
    for i in range(num_iterations):
        #Grab a batch
        X_batch = X[:,i*batch_size:i*batch_size+batch_size]
        Y_batch = Y[:,i*batch_size:i*batch_size+batch_size]

        #Forward Propogation
        h1 = sigmoid(np.matmul(W1.transpose(),X_batch) + b1)
        op = np.matmul(W2.transpose(),h1) + b2

        #Error Calculation
        error = mse(op, Y_batch)

        #Gradients Computation
        op_grad = np.subtract(op, Y_batch)/batch_size

        W2_grad = np.matmul(h1,op_grad.transpose())

        b2_grad = np.sum(op_grad,axis=1).reshape(b2.shape)

        h1_grad = np.matmul(W2,op_grad)

        W1_grad = np.matmul(X_batch, (h1_grad*h1*(1-h1)).transpose())

        b1_grad = np.sum(h1_grad * h1 * (1-h1),axis=1).reshape(b1.shape)


        #Update the last gradients

        W2_prev_grad = momentum * W2_prev_grad + (1-momentum) * W2_grad

        W1_prev_grad = momentum * W1_prev_grad + (1-momentum) * W1_grad

        b2_prev_grad = momentum * b2_prev_grad + (1-momentum) * b2_grad

        b1_prev_grad = momentum * b1_prev_grad + (1-momentum) * b1_grad


        #Weights Updation

        W2 -= W2_prev_grad

        W1 -= W1_prev_grad

        b2 -= b2_prev_grad

        b1 -= b1_prev_grad

    #Report the training error every 50 epochs

    if j % 50 == 0:
        h1 = sigmoid(np.matmul(W1.transpose(),X) + b1)
        op = np.matmul(W2.transpose(),h1) + b2
        error = mse(op, Y)
        err_momentum.append(error)
        print("Epoch {}: Error : {}".format(j,error))

#Training with Gradient Descent With AdaGrad

#Weights and biases initialisation
W1 = 2*np.random.rand(n_input,n_hidden)-1
b1 = 2*np.ones(shape=(n_hidden,1))-1
W2 = 2*np.random.rand(n_hidden,n_output)-1
b2 = 2*np.ones(shape=(n_output,1))-1

#Gradients initialisation
wg1 = np.ndarray(shape = W1.shape)
wg2 = np.ndarray(shape = W2.shape)
bg1 = np.ndarray(shape = b1.shape)
bg2 = np.ndarray(shape = b2.shape)

epsilon = np.exp(-8)
W2_cache = np.zeros(W2.shape) + epsilon
b2_cache = np.zeros(b2.shape) + epsilon
W1_cache = np.zeros(W1.shape) + epsilon
b1_cache = np.zeros(b1.shape) + epsilon
err_adagrad = []

for j in range(num_epochs):
    for i in range(num_iterations):
        #Grab a batch
        X_batch = X[:,i*batch_size:i*batch_size+batch_size]
        Y_batch = Y[:,i*batch_size:i*batch_size+batch_size]

        #Forward Propogation
        h1 = sigmoid(np.matmul(W1.transpose(),X_batch) + b1)
        op = np.matmul(W2.transpose(),h1) + b2

        #Error Calculation
        error = mse(op, Y_batch)

        #Gradients Computation
        op_grad = np.subtract(op, Y_batch)/batch_size

        W2_grad = np.matmul(h1,op_grad.transpose())

        b2_grad = np.sum(op_grad,axis=1).reshape(b2.shape)

        h1_grad = np.matmul(W2,op_grad)

        W1_grad = np.matmul(X_batch, (h1_grad*h1*(1-h1)).transpose())

        b1_grad = np.sum(h1_grad * h1 * (1-h1),axis=1).reshape(b1.shape)

        #Weights Updation

        W2 -= learning_rate * np.true_divide(W2_grad,np.sqrt(W2_cache))

        W1 -= learning_rate * np.true_divide(W1_grad,np.sqrt(W1_cache))

        b2 -= learning_rate * np.true_divide(b2_grad,np.sqrt(b2_cache))

        b1 -= learning_rate * np.true_divide(b1_grad,np.sqrt(b1_cache))

        #Update the cache

        W2_cache += W2_grad ** 2

        W1_cache += W1_grad ** 2

        b2_cache += b2_grad ** 2

        b1_cache += b1_grad ** 2

    #Report the training error every 50 epochs

    if j % 50 == 0:
        h1 = sigmoid(np.matmul(W1.transpose(),X) + b1)
        op = np.matmul(W2.transpose(),h1) + b2
        error = mse(op, Y)
        err_adagrad.append(error)
        print("Epoch {}: Error : {}".format(j,error))

#Training with Gradient Descent With AdaDelta/RMSProp

#Weights and biases initialisation
W1 = 2*np.random.rand(n_input,n_hidden)-1
b1 = 2*np.ones(shape=(n_hidden,1))-1
W2 = 2*np.random.rand(n_hidden,n_output)-1
b2 = 2*np.ones(shape=(n_output,1))-1

#Gradients initialisation
wg1 = np.ndarray(shape = W1.shape)
wg2 = np.ndarray(shape = W2.shape)
bg1 = np.ndarray(shape = b1.shape)
bg2 = np.ndarray(shape = b2.shape)

epsilon = np.exp(-8)
W2_cache = np.zeros(W2.shape) + epsilon
b2_cache = np.zeros(b2.shape) + epsilon
W1_cache = np.zeros(W1.shape) + epsilon
b1_cache = np.zeros(b1.shape) + epsilon
err_adadelta = []
decay_const = 0.9

for j in range(num_epochs):
    for i in range(num_iterations):
        #Grab a batch
        X_batch = X[:,i*batch_size:i*batch_size+batch_size]
        Y_batch = Y[:,i*batch_size:i*batch_size+batch_size]

        #Forward Propogation
        h1 = sigmoid(np.matmul(W1.transpose(),X_batch) + b1)
        op = np.matmul(W2.transpose(),h1) + b2

        error = mse(op, Y_batch)

        #Gradients Computation
        op_grad = np.subtract(op, Y_batch)/batch_size

        W2_grad = np.matmul(h1,op_grad.transpose())

        b2_grad = np.sum(op_grad,axis=1).reshape(b2.shape)

        h1_grad = np.matmul(W2,op_grad)

        W1_grad = np.matmul(X_batch, (h1_grad*h1*(1-h1)).transpose())

        b1_grad = np.sum(h1_grad * h1 * (1-h1),axis=1).reshape(b1.shape)

        #Weights Updation
        #Error Calculation

        W2 -= learning_rate * np.true_divide(W2_grad,np.sqrt(W2_cache))

        W1 -= learning_rate * np.true_divide(W1_grad,np.sqrt(W1_cache))

        b2 -= learning_rate * np.true_divide(b2_grad,np.sqrt(b2_cache))

        b1 -= learning_rate * np.true_divide(b1_grad,np.sqrt(b1_cache))

        #Update the cache

        W2_cache = decay_const * W2_cache + (1-decay_const) * W2_grad ** 2

        W1_cache = decay_const * W1_cache + (1-decay_const) * W1_grad ** 2

        b2_cache = decay_const * b2_cache + (1-decay_const) * b2_grad ** 2

        b1_cache = decay_const * b1_cache + (1-decay_const) * b1_grad ** 2

    #Report the training error every 50 epochs

    if j % 50 == 0:
        h1 = sigmoid(np.matmul(W1.transpose(),X) + b1)
        op = np.matmul(W2.transpose(),h1) + b2
        error = mse(op, Y)
        err_adadelta.append(error)
        print("Epoch {}: Error : {}".format(j,error))

#Training with Gradient Descent With Adam

#Weights and biases initialisation
W1 = 2*np.random.rand(n_input,n_hidden)-1
b1 = 2*np.ones(shape=(n_hidden,1))-1
W2 = 2*np.random.rand(n_hidden,n_output)-1
b2 = 2*np.ones(shape=(n_output,1))-1

#Gradients initialisation
wg1 = np.ndarray(shape = W1.shape)
wg2 = np.ndarray(shape = W2.shape)
bg1 = np.ndarray(shape = b1.shape)
bg2 = np.ndarray(shape = b2.shape)

epsilon = np.exp(-8)
W2_second_moment = np.zeros(W2.shape) + epsilon
b2_second_moment = np.zeros(b2.shape) + epsilon
W1_second_moment = np.zeros(W1.shape) + epsilon
b1_second_moment = np.zeros(b1.shape) + epsilon
W2_first_moment = np.zeros(W2.shape) + epsilon
b2_first_moment = np.zeros(b2.shape) + epsilon
W1_first_moment = np.zeros(W1.shape) + epsilon
b1_first_moment = np.zeros(b1.shape) + epsilon

err_adam = []
beta2 = 0.999
beta1 = 0.9

for j in range(num_epochs):
    for i in range(num_iterations):
        #Grab a batch
        X_batch = X[:,i*batch_size:i*batch_size+batch_size]
        Y_batch = Y[:,i*batch_size:i*batch_size+batch_size]

        #Forward Propogation
        h1 = sigmoid(np.matmul(W1.transpose(),X_batch) + b1)
        op = np.matmul(W2.transpose(),h1) + b2

        #Error Calculation
        error = mse(op, Y_batch)

        #Gradients Computation
        op_grad = np.subtract(op, Y_batch)/batch_size

        W2_grad = np.matmul(h1,op_grad.transpose())

        b2_grad = np.sum(op_grad,axis=1).reshape(b2.shape)

        h1_grad = np.matmul(W2,op_grad)

        W1_grad = np.matmul(X_batch, (h1_grad*h1*(1-h1)).transpose())

        b1_grad = np.sum(h1_grad * h1 * (1-h1),axis=1).reshape(b1.shape)

        #Bias_Correction Applied

        W2_first_moment_bias_corrected = W2_first_moment / (1 - beta1**(j+1))
        W1_first_moment_bias_corrected = W1_first_moment / (1 - beta1**(j+1))
        b2_first_moment_bias_corrected = b2_first_moment / (1 - beta1**(j+1))
        b1_first_moment_bias_corrected = b1_first_moment / (1 - beta1**(j+1))

        W2_second_moment_bias_corrected = W2_second_moment / (1 - beta2**(j+1))
        W1_second_moment_bias_corrected = W1_second_moment / (1 - beta2**(j+1))
        b2_second_moment_bias_corrected = b2_second_moment / (1 - beta2**(j+1))
        b1_second_moment_bias_corrected = b1_second_moment / (1 - beta2**(j+1))

        #Weights Updation

        W2 -= learning_rate * np.true_divide(W2_first_moment_bias_corrected,np.sqrt(W2_second_moment_bias_corrected + epsilon))
        W1 -= learning_rate * np.true_divide(W1_first_moment_bias_corrected,np.sqrt(W1_second_moment_bias_corrected + epsilon))
        b2 -= learning_rate * np.true_divide(b2_first_moment_bias_corrected,np.sqrt(b2_second_moment_bias_corrected + epsilon))
        b1 -= learning_rate * np.true_divide(b1_first_moment_bias_corrected,np.sqrt(b1_second_moment_bias_corrected + epsilon))

        #Update the moments

        W2_first_moment = beta1 * W2_first_moment + (1-beta1) * W2_grad

        W1_first_moment = beta1 * W1_first_moment + (1-beta1) * W1_grad

        b2_first_moment = beta1 * b2_first_moment + (1-beta1) * b2_grad

        b1_first_moment = beta1 * b1_first_moment + (1-beta1) * b1_grad

        W2_second_moment = beta2 * W2_second_moment + (1-beta2) * W2_grad ** 2

        W1_second_moment = beta2 * W1_second_moment + (1-beta2) * W1_grad ** 2

        b2_second_moment = beta2 * b2_second_moment + (1-beta2) * b2_grad ** 2

        b1_second_moment = beta2 * b1_second_moment + (1-beta2) * b1_grad ** 2

    #Report the training error every 50 epochs

    if j % 50 == 0:
        h1 = sigmoid(np.matmul(W1.transpose(),X) + b1)
        op = np.matmul(W2.transpose(),h1) + b2
        error = mse(op, Y)
        err_adam.append(error)
        print("Epoch {}: Error : {}".format(j,error))


#Plot the error for a better visualisation
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.plot(err,'b',label="SGD")
plt.plot(err_momentum,'r',label="SGD With Momentum")
plt.plot(err_adagrad,'g',label="AdaGrad")
plt.plot(err_adadelta,'black',label="AdaDelta/RMSProp")
plt.plot(err_adam,'orange',label="Adam")

plt.legend()
