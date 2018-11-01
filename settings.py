# coding=utf-8
import os
import sys

__ver__ = 0.2
__github__ = "tearsforyears"

'''
Note:
    this is the neural networks parameters settings
    the neural networks do not use for loop to generate
    so if you want to change your neural networks construct
    please use api conv_layer and fc_layer to rebuild
    
    ### do not change the construct of the neural net work!
'''

# append the project root to this file
# BASE_DIR = os.getcwd() in this file
BASE_DIR = r'F:/resourcefile/mnist_cnn'
sys.path.append(BASE_DIR)

# about dataset
DATA_PATH = BASE_DIR + r'/data'
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
DECAY_STEPS = 500
REG_RATE = 0.001
STEPS = 30000
DROPOUT_RATE = 0.5
MOVING_AVERAGE_DECAY = 0.99

# about model and save
MODEL_SAVE_PATH = BASE_DIR + r"/savedata/"
MODEL_NAME = 'mnist_cnn.ckpt'
MAX_TO_KEEP = 3
KEEP_CHECKPOINT_EVERY_N_HOURS = 1
PB_NAME = r'mnist_cnn.pb'

# tensorboard analysis
SUMMARY_DATA_PATH = BASE_DIR +r'/savedata/tensorboard/'