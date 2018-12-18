# coding=utf8
import tensorflow as tf
import numpy as np


def conv_block(inputs, filters_unm, kernels, strides, use_bias):
    """
    :param use_bias:
    :param inputs: shape=[batch_size,w,h,c]
    :param filters_unm: int
    :param kernels: w*h
    :param strides: w*h

    :return:
    """
    if not use_bias:
        x = tf.layers.conv2d(inputs=inputs, filters=filters_unm, kernel_size=kernels, strides=strides, padding="same",
                             use_bias=use_bias)
        x = tf.layers.batch_normalization(x)
    else:
        x = tf.layers.conv2d(inputs=inputs, filters=filters_unm, kernel_size=kernels, strides=strides, padding="same",
                             use_bias=use_bias)
    x = tf.nn.leaky_relu(x, alpha=0.2)

    return x


def deconv_block(inputs, filters_unm, kernels, strides):
    """
    :param use_bias:
    :param inputs: shape=[batch_size,w,h,c]
    :param filters_unm: int
    :param kernels: w*h
    :param strides: w*h

    :return:
    """

    x = tf.layers.conv2d_transpose(inputs=inputs, filters=filters_unm, kernel_size=kernels, strides=strides,
                                   padding="same", use_bias=False)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)

    return x


def down_block(x, filters_num, block_num):
    """
    :param x: input_Data
    :param filters_num: output channel
    :param block_num: the num of down_block
    :return:
    """
    with tf.name_scope("down_block_scale" + str(block_num)):
        x = conv_block(x, filters_unm=filters_num, kernels=(3, 3), strides=(1, 1), use_bias=False)
        x = conv_block(x, filters_unm=filters_num, kernels=(3, 3), strides=(2, 2), use_bias=False)
        return x


def up_block(x, filters_num, block_num):
    """
    :param x: input_Data
    :param filters_num: output channel
    :param block_num: the num of down_block
    :return:
    """
    with tf.name_scope("up_block_scale" + str(block_num)):
        x = deconv_block(x, filters_unm=filters_num, kernels=(3, 3), strides=(1, 1))
        x = deconv_block(x, filters_unm=filters_num, kernels=(3, 3), strides=(2, 2))
        return x


def u_net(input_data):
    """u_net net realize"""
    with tf.name_scope("encoder"):
        down1 = down_block(input_data, filters_num=64, block_num=1)
        down2 = down_block(down1, filters_num=128, block_num=2)
        down3 = down_block(down2, filters_num=256, block_num=3)
        down4 = down_block(down3, filters_num=512, block_num=4)

    with tf.name_scope("feature_map"):
        x = conv_block(down4, filters_unm=1024, kernels=(3, 3), strides=(1, 1), use_bias=False)
        x = conv_block(x, filters_unm=1024, kernels=(3, 3), strides=(1, 1), use_bias=False)

    with tf.name_scope("decoder"):
        up4 = up_block(x, filters_num=512, block_num=4)
        up4 = tf.concat([up4, down3], 3)

        up3 = up_block(up4, filters_num=256, block_num=3)
        up3 = tf.concat([up3, down2], 3)

        up2 = up_block(up3, filters_num=128, block_num=2)
        up2 = tf.concat([up2, down1], 3)

        up1 = up_block(up2, filters_num=64, block_num=1)
        up1 = tf.concat([up1, input_data], 3)

    with tf.name_scope("head"):
        x = conv_block(up1, filters_unm=64, kernels=(3, 3), strides=(2, 2), use_bias=True)  # 64*216
        x = conv_block(x, filters_unm=128, kernels=(3, 3), strides=(2, 2), use_bias=False)
        x = conv_block(x, filters_unm=256, kernels=(3, 3), strides=(2, 2), use_bias=True)
        x = conv_block(x, filters_unm=512, kernels=(3, 3), strides=(2, 2), use_bias=False)  # 8 * 32 * 512

    with tf.name_scope("fc"):
        x = tf.layers.conv2d(x, filters=1, kernel_size=(8, 13), strides=(1, 1), padding='valid',
                             use_bias=False)  # batch*1 * 20 * 1
        # x = tf.nn.softmax(x)
        x = x[:, 0, :, 0]
        return x


def eva_loss(label, predict):
    """eval loss"""
    # predict = tf.
    loss = tf.losses.mean_squared_error(labels=label, predictions=predict)
    all_vars = tf.trainable_variables()
    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
                        if 'bias' not in v.name]) * 0.001
    return loss + loss_l2
