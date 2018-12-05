# coding=utf8

import tensorflow as tf
import random
import cv2
import os
import numpy as np


class UNET:
    def __init__(self):
        self.image_path = 'train.txt'
        self.batch_size = 2

        self.image_size = [500, 100]
        self.label_size = 20

        self.num = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)

        # load data
        with open(self.image_path) as f:
            self.lines = f.readlines()
        random.shuffle(self.lines)

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
            img_set.append(img)
            label_set.append(list(map(eval, list(label))))
        self.num += self.batch_size
        return img_set, label_set

    def loss(self):

        x = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        y = tf.placeholder(tf.float32, [self.batch_size, self.label_size])


        print(23)


if __name__ == '__main__':
    u = UNET()
