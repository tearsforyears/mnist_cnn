# coding=utf-8
from preprocessing.load import load
import tensorflow as tf


def inference():
    pass


def main():
    print("a")


if __name__ == '__main__':
    x, y = load()
    print(x.shape, y.shape)
    import time
    time.sleep(500)