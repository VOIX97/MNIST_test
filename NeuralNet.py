#! /user/bin/env python

from load_samples import *
import numpy as np
import random
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

class NeuralNet(object):
  # initialize the network
  def __init__(self, sizes):
    self.sizes_ = sizes
    self.num_layers_ = len(sizes)
    self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]

  # sigmoid
  def sigmoid(self, z):
## question 1 —————————————————————— start
    return 1 / (1 + np.exp(-z))


## ——————————————————————————————————— end
  
  # derived function of sigmoid
  def sigmoid_prime(self, z):
## question 2 —————————————————————— start
    return self.sigmoid(z) * (1 - self.sigmoid(z))


## ——————————————————————————————————— end

  # feedforward
  def feedforward(self, x):
## question 3 —————————————————————— start



## ——————————————————————————————————— end


  # backprop
## ———————————————— explain the code below
  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.b_]
    nabla_w = [np.zeros(w.shape) for w in self.w_]

    activation = x
    activations = [x]
    zs = []

    for b, w in zip(self.b_, self.w_):
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = self.sigmoid(z)
      activations.append(activation)

    delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers_):
      z = zs[-l]
      sp = self.sigmoid_prime(z)
      delta = np.dot(self.w_[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
## ——————————————————————————————————— end

  # eta: learning rate
  def update_mini_batch(self, mini_batch, eta):
    nabla_b = [np.zeros(b.shape) for b in self.b_]
    nabla_w = [np.zeros(w.shape) for w in self.w_]
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.w_ = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.w_, nabla_w)]
    self.b_ = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.b_, nabla_b)]

  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data:
      n_test = len(test_data)

    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
      else:
        print("Epoch {0} complete".format(j))

  def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

  def cost_derivative(self, output_activations, y):
    return (output_activations-y)/(output_activations*(1-output_activations))

  # prediction
  def predict(self, data):
    value = self.feedforward(data)
    return value.tolist().index(max(value))
