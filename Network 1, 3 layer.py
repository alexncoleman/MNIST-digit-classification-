#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 11:09:43 2022

Code from: Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015
"""





#REMEMBER, weights depend of number of conections BETWEEN layers, and biases depend on the number of neurons.

import numpy as np
import random


#%%%%% Class and functions needed for basic network

class Network(object):
    
    def __init__(self, sizes):
        
        #The list ``sizes`` contains the number of neurons in the
        #respective layers of the network.  For example, if the list
        #was [2, 3, 1] then it would be a three-layer network, with the
        #first layer containing 2 neurons, the second layer 3 neurons,
        #and the third layer 1 neuron.
        
        
        
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #np.random.randn = random value with Gaussian PDF with mean of 0 and variance of 1. 
        #(cont. from above) number of parameters = dimension of array, and value of parameter = size of array
        #For biases only need one value per neuron in every layer (besides first)
        
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes[:-1], sizes[1:])] #zip creates tuple
        #value of x tells you number of neurons in each layer besides the last layer because you want to find number of connections for every neuron besides those in last layer
        #value of y tells you the number of connections each neuron has in each layer besides the first because nothing connects to the first layer.
        


    def feedforward(self, a): #returns output of network for input a 
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        #Train neural network with this function, using mini-batch stochastic gradient descent
        #The 'training_data' is a list of tuples '(x,y)' representing the training inputs and desired outputs
        #The other non-optional parameters are self explanatory (epochs = num. of times training data used), (eta is the learning rate)
        #IF 'test_data' provided then after each epoch the code will be tested with test_data and will print this out
        #^ This could be useful for tracking progress but will slow things down a LOT
        
        if test_data: 
            test_data = list(test_data) #turns zip to list
            n_test = len(test_data) #i.e. if data provided number of tests = length of 'test_data'
        
        training_data = list(training_data) #turns zip to list
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
        
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in range(0, n, mini_batch_size)]
            #mini_batches determined by selecting part of training data (after its randomly shuffled)
            #the batches are taken from k to k + mini_batch_size as we want them to have length mini_batch_size
            #k is a value that changes from 0 to the length of the training data (n) in steps of length mini_batch_size
        
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        
            if test_data:
                print('Epoch{}: {} / {}'.format(j, self.evaluate(test_data), n_test)) #uses func. evaluate defined below
        
            else:
                print('Epoch {0} complete'.format(j))
            
    def update_mini_batch(self, mini_batch, eta):
                #updates the network's weights and biases by applying gradient descent using backpropagation to a single mini batch
                #The mini batch is a list of tuples '(x,y)' and eta is the learning rate
    
        nabla_b = [np.zeros(b.shape) for b in self.biases]#make array of all zeros that is the same shape as array of biases, can then keep track of steps
        nabla_w = [np.zeros(w.shape) for w in self.weights]#same as above but array is same shape as weights, can then keep track of steps
    
        #nabla_b, nabla_w are the arrays for the sum of all the steps in gradient descent taken across the mini batch
        #delta_nabla_b, delta_naabla_w are the steps for one piece of data in the mini batch
    
    
    
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) #uses func. backprop (defined below)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #add the step to array that keeps track of all the steps
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        
        #now change value of weights and biases before next mini batch
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        #backprop given x and y where x is first layer activations and y is output layer values wanted
       #returns a tuple '(nabla_b, nabla_w)' (memory in functions are local)
       #tuple returned represents gradient for cost function C_x
       #'nabla_b' and 'nabla_w' are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
       
        nabla_b = [np.zeros(b.shape) for b in self.biases]#make array of all zeros that is the same shape as array of biases, can then keep track of steps
        nabla_w = [np.zeros(w.shape) for w in self.weights]#same as above but array is same shape as weights, can then keep track of steps
       
        #feedforward
        activation = x
        activations = [x] #list to store all activations, layer-by-layer
        zs = [] #list to store all z vectors, layer by layer
        
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b 
            #in 2,3,1 network example the first w provided will be an array of 3,2 whereas the activation
            #will just be a two dimensional array. However, when dotted the first activation will be multiplied
            #with the first array of weights (single activation value multiplied to all weights), corresponding to the activation of the first neuron and the weights it has
            #2nd activation with 2nd array of weights then all three arrays (all with 3 values) are added (two from dot and 1 from biases)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) #uses final activations and desired output activations to find cost, then finds derivative of this
        nabla_b[-1] = delta #how the final biases should change is just proportional to above
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #how weights should change depends on activations in second to last layer as well. also if differentated sigmoid wrt weights than change must be
        #multplied by activations as in the formula weights and activations are multiplied together
    
        #Note that the variable l is used different to the notation in chapter 2 of the book
        #here l = 1 means last layer of neurons, and l = 2 means second to last
        #its a renumbering to take advantage of negative indices in python
        for l in range(2, self.num_layers): #already carried out this process when l = 1 above so start from l =2
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp #to figure out how other layers change must use transposed weights to go 'backward' in network (as layout of the weights matrix has neurons in layer l as the columns and neurons in layer l+1 as the rows) , than multiply by sp again as it is another derivative
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return(nabla_b,nabla_w)

    def evaluate(self, test_data):
        """return the number of test inputs that the neural network
        outputs the correct results. The networks output is assumed to
        be whichever neuron in the final layer has the highest activation"""
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results) #checks how many neural network outputs (x) are equal to the desired output (y)

    def cost_derivative(self, output_activation, y):
        """Return the vector of the partial derivative partial C_x/partial a 
        for the output activations"""
        return (output_activation - y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    #derivative of sigmoid function
    return sigmoid(z)*(1-sigmoid(z))

#%%%% MNIST loader
     #### Libraries
# Standard library
import _pickle as cPickle
import gzip


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding = 'latin1' )
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#%%%%%% carrying out the loading and using functions

training_data, validation_data, test_data = load_data_wrapper()
   
net = Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data = test_data)


 