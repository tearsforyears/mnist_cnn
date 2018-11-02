# coding=utf-8
import tensorflow as tf
import numpy as np
from settings import *
from model.inference import forward_prop
from preprocessing.load import load, load_validate


def rebuild_by_ckpt(sess, saver):
    '''
    Note:
        to restore the model by ckpt
        if could use before define to rebuild the model
        and then you could use this model to implements checkpoint-learning
    '''
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("first train of this model")


def predict(input_x, use_probability=True):
    """
    Args:
         input_x: the data x to predict y
         use_probability: the predict labels type
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEP])
    y = forward_prop(x)
    saver = tf.train.Saver()

    # cast probability to labels
    predict_label = tf.argmax(y, 1)

    prediction = None
    with tf.Session() as sess:
        rebuild_by_ckpt(sess, saver)
        if use_probability:
            prediction = sess.run(y, feed_dict={x: input_x})
        else:
            prediction = sess.run(predict_label, feed_dict={x: input_x})
    return prediction


def accuracy(input_x, input_y):
    """
    Args:
         input_x: the data x to predict y
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEP])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, RESULT_KIND])
    y = forward_prop(x)
    saver = tf.train.Saver()

    # cast probability to labels
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = None
    with tf.Session() as sess:
        rebuild_by_ckpt(sess, saver)
        result = sess.run(accuracy, feed_dict={x: input_x, y_: input_y})
    return result


def main():
    import time
    x, y = load_validate()
    print(x.shape, y.shape)
    tic = time.time()
    result = accuracy(x, y)
    tok = time.time()
    print(result)
    print((tok - tic), 's')


if __name__ == '__main__':
    main()
