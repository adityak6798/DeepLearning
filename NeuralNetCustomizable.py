# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:52:47 2018

@author: Aditya Khandelwal
"""
import numpy as np
import theano.tensor as T
import theano
from theano.tensor.nnet.bn import batch_normalization_test, batch_normalization_train
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams

class NeuralNetwork:
    def __init__(self, architecture, dropout = None, W_init_type = 'standard_normal', batch_norm = False, activation = None, classification = False):
        self.X = T.dmatrix('X')
        self.layers = []
        self.layer_shapes = []
        if dropout == None:
            self.dropout = [0.0 for item in architecture]
        else:
            self.dropout = dropout
        if classification == False:
            for index in range(1,len(architecture)):
                h = HiddenLayer(architecture[index-1], architecture[index], W_init_type, batch_norm, activation, self.dropout[index-1])
                self.layers.append(h)
        else:
            for index in range(1,len(architecture)-1):
                h = HiddenLayer(architecture[index-1], architecture[index], W_init_type, batch_norm, activation)
                self.layers.append(h)
            self.layers.append(HiddenLayer(architecture[index-1], architecture[index], W_init_type, batch_norm, activation = 'softmax'))    
        self.Y = T.dmatrix('Y')
        self.parameters = []
        self.norm_params = []
        self.n_norm_params = []
        self.momentum_params = []
        self.cache = []
        self.adamfm = []
        self.adamsm = []
        for l in self.layers:
            for p in l.trainable_params:
                self.parameters.append(p)
                
    def train(self, x_ip, y_ip, lr, optimizer, num_epochs, batch_size, momentum = 0.9, rmsprop_decay_rate = 0.9, adam_beta1 = 0.9, adam_beta2 = 0.999):
        self.output = self.layers[0].forward(self.X, True)
        for p in self.layers[0].norm_params:
            self.norm_params.append(p)
        for p in self.layers[0].n_norm_params:
            self.n_norm_params.append(p)    
        self.intermediate_vals = [self.output]
        for l in self.layers[1:]:
            self.intermediate_vals.append(l.forward(self.intermediate_vals[-1], True))            
            for p in l.norm_params:
                self.norm_params.append(p)
            for p in l.n_norm_params:
                self.n_norm_params.append(p)

        self.cache = [theano.shared(np.zeros_like(p.get_value())) for p in self.parameters]
        
        self.momentum_params = [theano.shared(np.zeros_like(p.get_value())) for p in self.parameters]

        self.adamfm = [theano.shared(np.zeros_like(p.get_value())) for p in self.parameters]

        self.adamsm = [theano.shared(np.zeros_like(p.get_value())) for p in self.parameters]

        self.loss = T.mean(T.sqr(self.Y - self.layers[-1].A))
        
        self.gradients = [T.grad(self.loss, p) for p in self.parameters]
        
        self.updates = [(p, np) for p, np in zip(self.norm_params, self.n_norm_params)]
        
        self.adamc = theano.shared(1)
        
        if optimizer == 'SGD':
            self.updates += [(p, p - lr * g) for p,g in zip(self.parameters,self.gradients)]
            
        elif optimizer == 'SGD with Momentum':
            self.updates += [(mp, - momentum*mp + lr*g) for mp,g in zip(self.momentum_params, self.gradients)]
            self.updates += [(p,p - mp) for p,mp in zip(self.parameters,self.momentum_params)]
            
        elif optimizer == 'AdaGrad':
            self.updates += [(c, c + T.sqr(g)) for c,g in zip(self.cache, self.gradients)]
            #print(self.updates)
            
            self.updates += [(p, p - (lr * g / T.sqrt(c + 1e-8))) for p, g, c in zip(self.parameters, self.gradients, self.cache)]
            #print(self.updates)
            
        elif optimizer == 'RMSProp':
            self.updates += [(c, rmsprop_decay_rate*c + (1-rmsprop_decay_rate)*T.sqr(g)) for c,g in zip(self.cache, self.gradients)]
            self.updates += [(p, p - lr * T.true_div(g,T.sqrt(c + 1e-8))) for p, g, c in zip(self.parameters, self.gradients, self.cache)]
            #self.updates += [(c, rmsprop_decay_rate*c + (1-rmsprop_decay_rate)*T.sqr(g)) for c,g in zip(self.cache, self.gradients)]
            
        elif optimizer == 'Adam':
            self.updates += [(p, p - lr * T.true_div(fm,T.sqrt(sm + 1e-8))) for p,fm,sm in zip(self.parameters, self.adamfm, self.adamsm)]
            self.updates += [(fm, adam_beta1*fm + (1-adam_beta1)*g) for fm, g in zip(self.adamfm, self.gradients)]
            self.updates += [(sm, adam_beta2*sm + (1-adam_beta2)*T.sqr(g)) for sm, g in zip(self.adamsm, self.gradients)]
            self.updates += [(self.adamc, self.adamc+1)]
            
        else:
            raise ValueError("Optimizer not correct")
        
        self.train_inputs = [self.X, self.Y]
        self.train_outputs = [self.loss]
        #print(self.updates)
        self.train_fn = theano.function(inputs = self.train_inputs, outputs = self.train_outputs, updates = self.updates)
        
        error_plot = []
        
        num_iter = x_ip.shape[0] // batch_size
        for epoch in range(num_epochs):
            for n_iter in range(num_iter):
                x_batch = x_ip[n_iter * batch_size:n_iter * batch_size + batch_size,:]
                y_batch = y_ip[n_iter * batch_size:n_iter * batch_size + batch_size,:]
                self.train_fn(x_batch, y_batch)
            if epoch % 2 == 0:
                error_plot.append(self.train_fn(x_ip, y_ip))
        plt.plot(error_plot)
        #print(error_plot)
        
    def predict(self, x_ip):
        
        self.X_pred = T.dmatrix('X_pred')
        self.output = self.layers[0].forward(self.X_pred, False)
        self.parameters = []
        for l in self.layers[1:]:
            self.output = l.forward(self.output, False)
        self.pred_ip = [self.X]
        self.pred_op = [self.output]
        self.pred_fn = theano.function(inputs = self.pred_ip, outputs = self.pred_op)
        return self.pred_fn(x_ip)
    
class HiddenLayer:
    def __init__(self, size0, size1, W_init_type, batch_normalization, activation, dropout_rate = 0.0):
        if W_init_type == 'standard_normal':
            self.weights = theano.shared(np.random.standard_normal((size0, size1)))
        elif W_init_type == 'glorot':
            self.weights = np.random.normal(scale=np.sqrt(2.0/(size0+size1)),size=(size0,size1))
        else:
            raise ValueError("Check type of Weight Initialisation")
        if batch_normalization == True:
            self.gamma = theano.shared(np.ones(size1))
            self.beta = theano.shared(np.zeros(size1))
            self.running_mean = theano.shared(np.zeros(size1))
            self.running_variance = theano.shared(np.zeros(size1))
            self.trainable_params = [self.weights, self.gamma, self.beta]
            self.norm_params = [self.running_mean, self.running_variance]
        else:
            self.biases = theano.shared(np.ones(size1))
            self.trainable_params = [self.weights, self.biases]
            self.norm_params = []
        self.batch_norm = batch_normalization
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.rng = RandomStreams()
    def forward(self, prev_layer, train):
        self.drop = self.rng.binomial(size = prev_layer.shape, p = 1 - self.dropout_rate)
        prev_layer = prev_layer * self.drop
        self.Z = T.dot(prev_layer,self.weights)
        if self.batch_norm == True:
            if train == True:
                self.Z, _, _, self.n_running_mean, self.n_running_variance = batch_normalization_train(self.Z, self.gamma, self.beta, running_mean=self.running_mean, running_var=self.running_variance)
                self.n_norm_params = [self.n_running_mean, self.n_running_variance]
            else:
                self.Z = batch_normalization_test(self.Z, self.gamma, self.beta, self.running_mean, self.running_variance)
        else:
            self.Z += self.biases
            self.n_norm_params = []
        if self.activation == 'relu':
            self.A = T.nnet.nnet.relu(self.Z)
        elif self.activation == 'sigmoid':
            self.A = T.nnet.nnet.sigmoid(self.Z)
        elif self.activation == 'tanh':
            self.A = 2*T.nnet.nnet.sigmoid(self.Z) - 1
        elif self.activation == 'leaky_relu':
            self.A = T.nnet.nnet.relu(self.Z,alpha=0.1)
        elif self.activation == 'softmax':
            self.A = T.nnet.nnet.softmax(self.Z)
        else:
            raise ValueError('Activation Error')
        return self.A
        
        
if __name__ == '__main__':
    n = NeuralNetwork([5,5,2], activation = 'relu', batch_norm = False, dropout = [0.2,0.3])
    n.train(np.random.randn(100,5),np.random.rand(100,2), lr = 0.001, optimizer = 'SGD with Momentum', num_epochs = 256, batch_size = 32)
