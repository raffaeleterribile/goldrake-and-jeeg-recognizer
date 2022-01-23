# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:56:35 2017

@author: Raffaele
"""

import numpy as np

values = np.array([3.4, 44, 42, 998.888])

print(values.ndim)
print(values.shape)
print(values.size)
print(values.dtype)
print(values.itemsize)
print(values.data)
#------------------------------------------------------------------------------

a = np.arange(15).reshape(3, 5)
print(a)

print(a.shape)

print(a.ndim)

print(a.dtype.name)

print(a.itemsize)

print(a.size)

print(type(a))

b = np.array([6, 7, 8])
print(b)

print(type(b))
#------------------------------------------------------------------------------
zeros = np.zeros((5, 4))
ones = np.ones((5, 4))
empty = np.empty((5, 4))
print(zeros)
print(ones)
print(empty)
print(np.array([[1 + 0.4j], [3.5 + 98j]]))
print(np.arange(23, 78, 5.5))
print(np.linspace(23, 78, 21))