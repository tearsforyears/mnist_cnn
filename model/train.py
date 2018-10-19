# coding=utf-8
import tensorflow as tf
import numpy as np
from settings import *
from .inference import forward_prop
from preprocessing.load import load

'''
    forward_prop(input, train=None, regularizer=None, dropout_rate=DROPOUT_RATE)
'''
def get_Regularizer_Term(name):
    '''
    Args:
        name : for the collection which store the weights
    Return:
        the Regularizer_Term
    '''
    return tf.contrib.layers.l2_regularizer(REG_RATE)


def train():
    x = tf.placeholder(name="input_x", dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEP])
    y_ = tf.placeholder(name="input_y", dtype=tf.float32, shape=[None, RESULT_KIND])
    # this is for train x and y
    regularizer = tf.contrib.layers.l2_regularizer(REG_RATE)

def main():
    x, y = load()


if __name__ == '__main__':
    main()
