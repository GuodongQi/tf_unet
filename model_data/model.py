# coding=utf8

import tensorflow as tf
import numpy as np


def conv_block(inputs, filters, kernels, strides, use_bias):
    """
    :param use_bias:
    :param inputs: shape=[batch_size,w,h,c]
    :param filters: int
    :param kernels: w*h
    :param strides: w*h

    :return:
    """
    x = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernels, strides=strides, padding="same",
                         use_bias=use_bias)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)

    return x


def encoder(x):
    with tf.name_scope("encoder"):
        with tf.name_scope("scale"):
            x = conv_block(x, filters=64, kernels=(3, 3), strides=())

    return
