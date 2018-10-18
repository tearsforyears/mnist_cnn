# coding=utf-8
from preprocessing.load import load
import tensorflow as tf
# constant define
from settings import *


def conv_layer(name, input, filter_shape, stride=None):
    '''
    Args:
        input is about m,width,height,channel-type data
        name is the scope name
        filter=[conv_size, conv_height, input_channel, output_channel]
    '''
    relu = None
    with tf.name_scope(name):
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
    '''
    # in order to get the variable more simply
    pool = None
    with tf.name_scope(name):
        if ksize == None and strides == None:
            ksize = [1, 2, 2, 1]
            strides = [1, 2, 2, 1]
        pool = tf.nn.max_pool(input,
                              ksize=ksize,
                              strides=strides,
                              padding="SAME")
    return pool


def fc_layer(name, input, input_nodes, output_nodes):
    '''
    Args:
        input is a vector m,input nodes
    '''
    with tf.name_scope(name):
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
    return tf.nn.relu(z)


def forward():
    pass


def main():
    print("a")


if __name__ == '__main__':
    pass
