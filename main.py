#! /user/bin/env python

from load_samples import *
from NeuralNet import *
import numpy as np
import random
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

def MINST():
  INPUT = 28*28
  OUTPUT = 10
  net = NeuralNet([INPUT, 40, OUTPUT])

  train_set = load_samples(dataset='training_data')
  test_set = load_samples(dataset='testing_data')

  net.SGD(train_set, 13, 100, 3.0, test_data=test_set)
  
  # accuracy
  correct = 0;
  for test_feature in test_set:
    if net.predict(test_feature[0]) == test_feature[1][0]:
      correct += 1
  print("准确率: ", correct/len(test_set))



if __name__ == '__main__':
    MINST()

