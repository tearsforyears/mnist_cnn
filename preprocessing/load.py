# coding=utf-8
import os
import struct
import numpy as np
import cv2
# constant define
from settings import *


def load_mnist(path=DATA_PATH, kind='train'):
    # the code in mnist sample to load origin data set
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)  # %的作用是代替那个占位符
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def reformat(images, labels, batch_size=None):
    # change x to m,h,w,c or NHWC
    # change y to one-hot encoding
    if batch_size == None:
        x_ = images.reshape(DATA_SET_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEP)
    else:
        x_ = images.reshape(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEEP)
    y_ = np.eye(RESULT_KIND)[labels]  # one-hot encoding
    return x_, y_


def normalize(x):
    # according to mnist feature use 0,255->0,1 not (x-mean)/std
    # this is only we call scaling of dataset
    x = x / (np.max(x) - np.min(x))
    # python3 division is about float
    return x


def distribution(images, labels):
    import matplotlib.pyplot as plt
    # plot the images and labels distribution
    plt.subplot(121)
    plt.hist(images.reshape(-1))
    plt.subplot(122)
    plt.hist(labels)
    plt.show()


def inspect(images=None, labels=None, index=None, delay=2000):
    # in order make more convince to test
    if not (images and labels):
        images, labels = load_mnist()
    # check the data of image
    # print("the images shape is", images.shape)
    # print("the label shape is", labels.shape)

    if index == None:
        index = np.random.randint(0, DATA_SET_SIZE)
    cv2.imshow("{index} image with label:{label}".format(index=index, label=labels[index]),
               cv2.resize(images[index].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), (400, 400))  # 灰度图
               )
    # check the random sample
    cv2.waitKey(delay)


def load(images=None, labels=None):
    if not (images and labels):
        images, labels = load_mnist()
    x, y = reformat(images, labels)
    x = normalize(x)
    return x, y


def load_validate(images=None, labels=None):
    if not (images and labels):
        images, labels = load_mnist(kind='t10k')
    x, y = reformat(images, labels, 10000)
    x = normalize(x)
    return x, y


def get_start_and_end(index):
    start = (index * BATCH_SIZE) % DATA_SET_SIZE
    end = min(start + BATCH_SIZE, DATA_SET_SIZE)
    return start, end


def get_batch(input, index=0):
    start, end = get_start_and_end(index)
    return input[start:end]


def get_label(label, index=0):
    start, end = get_start_and_end(index)
    return label[start:end]


if __name__ == '__main__':
    images, labels = load_mnist()
    # distribution(images, labels)
    x, y = load()
    pass
