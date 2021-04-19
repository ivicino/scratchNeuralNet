#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:53:06 2021

@author: ivicino
"""

"""
We want to be able to have as many layers as we want, and of any type.
But if we modify/add/remove one layer from the network, the output of the
network is going to change, which is going to change the error, which is going
 to change the derivative of the error with respect to the parameters.
 We need to be able to compute the derivatives regardless of the network
 architecture, regardless of the activation functions, regardless of the loss
 we use.

In order to achieve that, we must implement each layer separately.

"""


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
