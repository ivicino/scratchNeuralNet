#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:48:48 2021

@author: ivicino
"""

import numpy as np

from NeuralNet import NeuralNet
from fclayer import FCLayer
# from activationlayer import ActivationLayer
# from activations import tanh, tanh_prime
from losses import mse, mse_prime


# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
# y_train = np.array([[[0]], [[0]], [[0]], [[1]]])    # AND logic gate - works
# y_train = np.array([[[1]], [[0]], [[0]], [[0]]])  # NOR logic gate - works
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])  # XNOR logic gate - NOPE!
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])    # XOR logic gate - NOPE!
# Gets stuck after ~126 iterations (epochs)

# network
net = NeuralNet()
net.add(FCLayer(2, 1))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.05)

# test
out = net.predict(x_train)
print(out)
