# coding=utf-8
import tensorflow as tf
import numpy as np
from settings import *
from .inference import forward_prop
from preprocessing.load import load

'''
    forward_prop(input, train=None, regularizer=None, dropout_rate=DROPOUT_RATE)
'''


def get_Regularizer_Term(name='loss'):
    '''
    Args:
        name : for the collection which store the weights
    Return:
        the Regularizer_Term
    '''
    return tf.add_n(tf.get_collection(name))


def get_learning_rate(global_step, decay=True):
    '''
    Args:
        learning_rate:
        global_step: the iteration step or current training step
    Return:
        the learning_rate with decays
    '''
    if decay:
        a = tf.train.exponential_decay(
            LEARNING_RATE,
            global_step=global_step,
            decay_steps=DECAY_STEPS,
            decay_rate=LEARNING_RATE_DECAY,
            staircase=False
        )
        return a
    else:
        return LEARNING_RATE


def loss_function(y_, y):
    '''
    Args:
        y: this is the predict y
        y_: this is the ground true label y_
    Return:
        the loss function's value
    Others:
        use cross_entropy to measure the loss
        of implement MLE or other functions here
    '''
    cross_entropy = -y_ * tf.log(tf.clip_by_value(y, 1e-8, 1.))
    loss = tf.reduce_mean(cross_entropy) + get_Regularizer_Term('loss')


def train(data_input, data_labels):
    # train data interface
    x = tf.placeholder(name="input_x", dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEP])
    y_ = tf.placeholder(name="input_y", dtype=tf.float32, shape=[None, RESULT_KIND])

    # regularizer and forward_prop
    regularizer = tf.contrib.layers.l2_regularizer(REG_RATE)
    y = forward_prop(x, True, regularizer)

    # training params: add_global learning_rate loss
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    learning_rate = get_learning_rate(global_step)
    loss = loss_function(y_, y)

    # optimizer the train (the core of the cnn)
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # get start with training
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % DATA_SET_SIZE
            end = min(start + BATCH_SIZE, DATA_SET_SIZE)
def main():
    images, labels = load()  # here the images is scaling and labels are one-hot encoding
    print(images.shape, labels.shape)


if __name__ == '__main__':
    main()
