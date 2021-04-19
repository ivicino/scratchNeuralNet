#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:11:10 2021

@author: ivicino
"""
# Why need non-linear activation functions:
# The decisions are taken by the non-linear layers by dropping data points that
# are less relevant than others

import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;
