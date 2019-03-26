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
    for b, w in zip(self.b_, self.w_):
      x = self.sigmoid(np.dot(w, x) + b)
    return x


## ——————————————————————————————————— end


  # backprop
## ———————————————— explain the code below
  def backprop(self, x, y):
    #初始化误差矩阵，用于储存各个层的weights和bias的误差
    nabla_b = [np.zeros(b.shape) for b in self.b_]
    nabla_w = [np.zeros(w.shape) for w in self.w_]
    #传递feature，将输入与各个层的输出均存储到activations中
    activation = x
    activations = [x]
    #zs存储的内容为activations存储的内容激活之前的值
    zs = []

    #取出现有每一层的bias和weights
    for b, w in zip(self.b_, self.w_):
      #从上一层的输入得到输出
      z = np.dot(w, activation)+b
      #存储该层的输出
      zs.append(z)
      #将输出用激活函数激活
      activation = self.sigmoid(z)
      #存储激活后的输出值
      activations.append(activation)

    #计算label与激活后的最终输出的差，并乘上最终输出的偏导数，得到最后一层的bias误差值
    delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
    #存储最后一层bias的误差值
    nabla_b[-1] = delta
    #将bias的误差与倒数第二层的激活后的输出矩阵的转置相乘，得到最后一层的weights的误差并存储
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    #开始从倒数第二层往前计算每一层的bias和weights误差
    for l in range(2, self.num_layers_):
      #取出倒数第L层的输出
      z = zs[-l]
      #计算倒数第L层在激活函数的偏导数，存为sp
      sp = self.sigmoid_prime(z)
      #计算倒数第L-1层的weights与倒数第L-1层的bias误差，计算点积并乘以sp，得到倒数第L层bias的误差值
      delta = np.dot(self.w_[-l+1].transpose(), delta) * sp
      #存储倒数第L层bias的误差值
      nabla_b[-l] = delta
      #将倒数第L层的bias的误差与倒数第L+1层的激活后的输出矩阵的转置相乘，得到倒数第L层的weights的误差并存储
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    #返回通过bp算法计算得的bias与weights的误差值
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
    for j in range(epoch
    s):
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
