# coding=utf-8
from preprocessing.load import load
import tensorflow as tf
from functools import reduce
from settings import *  # constant define
import numpy as np


def conv_layer(name, input, filter_shape, stride=None):
    '''
    Args:
        input is about m,width,height,channel-type data
        name is the scope name
        filter=[conv_size, conv_height, input_channel, output_channel]
    Return:
        relu is about 4-dim data
    '''
    relu = None
    with tf.name_scope(name):
        with tf.variable_scope(name):
            if stride == None:
                stride = [1, 1, 1, 1]
            conv_w = tf.get_variable(
                name='conv_w',
                shape=filter_shape,
                initializer=tf.truncated_normal_initializer(stddev=.1)
            )
            conv = tf.nn.conv2d(input, conv_w, stride, padding='SAME')
            conv_b = tf.get_variable(
                name='conv_b',
                shape=[filter_shape[-1]],
                initializer=tf.constant_initializer(.1)
            )
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_b))
    return relu


def pool_layer(name, input, ksize=None, strides=None):
    '''
    Args:
        input is about 4-dim data
        ksize for the size of filter
        strides for the strides of the filter
    Return:
        pool is about 4-dim data
    '''
    # in order to get the variable more simply
    pool = None
    with tf.name_scope(name):
        with tf.variable_scope(name):
            if ksize == None and strides == None:
                ksize = [1, 2, 2, 1]
                strides = [1, 2, 2, 1]
            pool = tf.nn.max_pool(input,
                                  ksize=ksize,
                                  strides=strides,
                                  padding="SAME")
    return pool


def fc_layer(name, input, input_nodes, output_nodes, regularizer=None):
    '''
    Args:
        input is a vector m,input nodes
    '''
    with tf.name_scope(name):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name="w",
                shape=[input_nodes, output_nodes],
                initializer=tf.truncated_normal_initializer(stddev=.1)
            )
            b = tf.get_variable(
                name="b",
                shape=[1, output_nodes],
                initializer=tf.truncated_normal_initializer(stddev=.1)
            )
            z = tf.matmul(input, w) + b
            if regularizer != None:
                tf.add_to_collection("reg", regularizer(w))  # regularization
    return tf.nn.relu(z)


def forward(input, train=None, regularizer=None, dropout_rate=DROPOUT_RATE):
    # conv layer and pool layer
    conv1 = conv_layer("conv1-layer", input, [CONV_1_HEIGHT, CONV_1_WIDTH, IMAGE_DEEP, CONV_1_DEEP])
    pool1 = pool_layer("pool1_layer", conv1)
    conv2 = conv_layer("conv2-layer", pool1, [CONV_2_HEIGHT, CONV_2_WIDTH, CONV_1_DEEP, CONV_2_DEEP])
    pool2 = pool_layer("pool2_layer", conv2)

    # the last pool to a vector(matrix) and do the fully connect job
    nodes = reduce(lambda x, y: x * y, pool2.get_shape().as_list()[1:4])
    input = tf.reshape(pool2, [-1, nodes])

    # fully connect layer
    fc1 = fc_layer("fc1-layer", input, nodes, FC_NODE_1, regularizer)
    if train: fc1 = tf.nn.dropout(fc1, dropout_rate)  # dropout
    fc2 = fc_layer("fc2-layer", fc1, FC_NODE_1, FC_NODE_2, regularizer)
    if train: fc2 = tf.nn.dropout(fc2, dropout_rate)  # dropout

    # softmax to predict probability
    return tf.nn.softmax(fc2)


def main():
    import time
    x = tf.constant(np.random.rand(20000, 28, 28, 1), tf.float32)  # fake test data
    y = forward(x)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tic = time.time()
        res = sess.run(y)
        tok = time.time()
        print(res.shape, tok - tic, 's')


if __name__ == '__main__':
    main()
