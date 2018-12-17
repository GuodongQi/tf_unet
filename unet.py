# coding=utf8

import tensorflow as tf
import random
from model_data.model import *
from utils.utils import *
import cv2
import os
import numpy as np


class UNET:
    def __init__(self):
        self.image_path = './data/train.txt'
        self.batch_size = 2

        self.image_size = [500, 100]
        self.label_size = 20
        self.epoch_num = 50

        self.num = 0  # the nth of line

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # load data
        with open(self.image_path) as f:
            self.lines = f.readlines()
        random.shuffle(self.lines)
        self.eval()

    def load_data(self):
        """generate a batch data"""
        img_set = []
        label_set = []
        for i in range(self.batch_size):
            if self.num + self.batch_size > len(self.lines):
                pdata = self.lines[len(self.lines) - i]
            else:
                pdata = self.lines[self.num + i]

            img, label = pdata.split()
            img_set.append("./data/picture/" + img + ".jpg")
            label_set.append(list(map(eval, list(label))))
        self.num += self.batch_size
        return img_set, label_set

    def eval(self):

        x = tf.placeholder(tf.float32, [self.batch_size, 512, 128, 3])
        # x = tf.placeholder(tf.float32, None)
        label = tf.placeholder(tf.float32, [self.batch_size, self.label_size])
        pred = u_net(x)
        loss = eva_loss(label, pred)

        opt = tf.train.AdamOptimizer()
        op = opt.minimize(loss)
        init = tf.global_variables_initializer()

        for i in range(self.epoch_num):
            self.num = 0
            for j in range(int(len(self.lines) / self.batch_size)):
                image_set, label_set = self.load_data()
                # label_set = tf.Variable(label_set,False)
                image_set = tf_read_image(image_set, sess=self.sess)
                pred_, loss_, _ = self.sess.run([init, pred, loss, op], feed_dict={x: image_set, label: label_set})
                print(pred_, loss_)


if __name__ == '__main__':
    UNET()
