# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 00:07:11 2018

@author: Raffaele
"""

import tensorflow as tf

a = tf.constant([8])
b = tf.constant([9])
sum = tf.add(a, b)

with tf.Session() as session:
    result = session.run(sum)
    print(result)
    
x = tf.Variable(4)
y = tf.Variable(3)
assign1 = tf.assign(x, 7)
assign2 = tf.assign(y, 17)
sum = tf.add(assign1, assign2)
variables_initializer = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(variables_initializer)
    result = session.run(sum)
    print(result)
    
p = tf.placeholder(tf.int16)
c1 = 30
c2 = 40
sum = tf.add(p, c1)
with tf.Session() as session:
    result1 = session.run(sum, feed_dict = {p: 3})
    result2 = session.run(sum, feed_dict = {p: 9})
    print(result1)
    print(result2)