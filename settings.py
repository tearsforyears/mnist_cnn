# coding=utf-8
import os
import sys

# append the project root to this file
# BASE_DIR = os.getcwd() in this file

BASE_DIR = r'F:\resourcefile\mnist_cnn'
sys.path.append(BASE_DIR)

# about dataset
DATA_PATH = BASE_DIR + r'\data'
DATA_SET_SIZE = 60000  # mnist 数据集大小
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_DEEP = 1
RESULT_KIND = 10

# about nerual networks
# conv
CONV_1_HEIGHT = 5
CONV_1_WIDTH = 5
CONV_1_DEEP = 32
CONV_2_HEIGHT = 5
CONV_2_WIDTH = 5
CONV_2_DEEP = 32
# fc layer
FC_NODE_1 = 512
FC_NODE_2 = 10

# about training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.95
REG_RATE = 0.0001
STEPS = 30000
DROPOUT_RATE = 0.5
