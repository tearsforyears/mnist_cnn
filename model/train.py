# coding=utf-8
import tensorflow as tf
import numpy as np
from settings import *
from model.inference import forward_prop
from preprocessing.load import load, get_label, get_batch
from tensorflow.python.framework import graph_util


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
    return loss


def variable_average(global_step):
    '''
    Note:
        ExponentialMovingAverage() to build a operation of MovingAverage with shadow variable
        ema.apply() to average the variable trainable_variables
    '''
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = ema.apply(tf.trainable_variables())
    return variables_averages_op


def save_model(sess, global_step, type='ckpt', saver=None):
    '''
    Args:
        type: ckpt or pb
        saver: need when only use ckpt to store the model
    Note:
        use tensorflow to save as a pb or a ckpt model
    '''
    if type == 'ckpt':
        # save as ckpt model
        saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME, global_step)
    elif type == 'pb':
        # save as pb model
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ["output"])
        with tf.gfile.GFile(MODEL_SAVE_PATH + PB_NAME, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    else:
        print("type error: do not save the model")  # use log is better


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


def backward_prop(data_input, data_labels):
    '''
    Args:
         data_input,data_labels the input of x and y after encoding
    Return:
        None
    Note:
        this is a function to implement the backward_prop of cnn
        use training data to train the model and save the model as
        ckpt or pb to get predict or to validate the model
    '''
    # train data interface
    x = tf.placeholder(name="input_x", dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEP])
    y_ = tf.placeholder(name="input_y", dtype=tf.float32, shape=[None, RESULT_KIND])

    # regularizer and forward_prop
    regularizer = tf.contrib.layers.l2_regularizer(REG_RATE)
    y = forward_prop(x, train=True, regularizer=regularizer)

    # global_step
    global_step = tf.Variable(-1, trainable=False)
    add_global = global_step.assign_add(1)

    # average the params
    average_op = variable_average(global_step)

    # learning_rate and loss function
    learning_rate = get_learning_rate(global_step)
    loss = loss_function(y_, y)

    # optimizer the train (the core of the cnn)
    optimize_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_op = tf.group(optimize_op, average_op)  # group this two operation to one

    # get start with training and save
    saver = tf.train.Saver(
        max_to_keep=MAX_TO_KEEP,
        keep_checkpoint_every_n_hours=KEEP_CHECKPOINT_EVERY_N_HOURS,
    )
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        rebuild_by_ckpt(sess, saver)  # break point
        for i in range(sess.run(global_step), STEPS):
            sess.run(
                train_op,
                feed_dict={
                    x: get_batch(data_input, index=i),
                    y_: get_label(data_labels, index=i)
                }
            )
            sess.run(add_global)
            if i % 2000 == 0:
                c = sess.run(loss, feed_dict={x: get_batch(data_input, i), y_: get_label(data_labels, i)})
                print("after training", i, "current loss become", c)
                save_model(sess, global_step, 'ckpt', saver)
                save_model(sess, global_step, 'pb')


def main():
    images, labels = load()  # here the images is scaling and labels are one-hot encoding
    print(images.shape, labels.shape)
    backward_prop(images, labels)


if __name__ == '__main__':
    main()
