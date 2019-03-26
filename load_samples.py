#! /user/bin/env python

import numpy as np
import random
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

def load_mnist(dataset="training_data", digits=np.arange(10), path="."):

  if dataset == "training_data":
    fname_image = os.path.join(path, 'MNIST_data/train-images-idx3-ubyte')
    fname_label = os.path.join(path, 'MNIST_data/train-labels-idx1-ubyte')
  elif dataset == "testing_data":
    fname_image = os.path.join(path, 'MNIST_data/t10k-images-idx3-ubyte')
    fname_label = os.path.join(path, 'MNIST_data/t10k-labels-idx1-ubyte')
  else:
    raise ValueError("dataset must be 'training_data' or 'testing_data'")
  print("Start loading ",dataset)
  flbl = open(fname_label, 'rb')
  magic_nr, size = struct.unpack(">II", flbl.read(8))
  lbl = pyarray("b", flbl.read())
  flbl.close()

  fimg = open(fname_image, 'rb')
  magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
  img = pyarray("B", fimg.read())
  fimg.close()

  ind = [ k for k in range(size) if lbl[k] in digits ]
  N = len(ind)
  
  images = zeros((N, rows, cols), dtype=uint8)
  labels = zeros((N, 1), dtype=int8)
  for i in range(len(ind)):
    images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
    labels[i] = lbl[ind[i]]
  
  return images, labels

def load_samples(dataset="training_data"):
  
  image,label = load_mnist(dataset)

  X = [np.reshape(x,(28*28, 1)) for x in image]
  X = [x/255.0 for x in X]
  
  def vectorized_Y(y): 
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

  if dataset == "training_data":
    Y = [vectorized_Y(y) for y in label]
    pair = list(zip(X, Y))
    return pair
  elif dataset == 'testing_data':
    pair = list(zip(X, label))
    return pair
  else:
    print('Something wrong')
