# coding=utf8

import tensorflow as tf
import cv2
import numpy as np

from model_data.model import u_net


class Unet:
    def __init__(self):
        # self.latest = "./model/checkpoint[0]"  # your checkpoint path
        self.latest = tf.train.latest_checkpoint("./model")  # your "checkpoint" path

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)

        self.x = tf.placeholder(tf.float32, [1, 80, 416, 3])

    def load_data(self):
        """read image"""
        inage_path = input("input image path:\n")
        im = cv2.imread(inage_path)

        assert im is not None, "open error, please check filename"

        im = im[:, 78:360 + 78, ::-1]  # BGR to RGB and crop
        im = cv2.resize(im, (416, 80))
        im = im / 255.0
        return np.stack([im])

    def predict(self):
        pred = u_net(self.x)

        saver = tf.train.Saver()
        saver.restore(self.sess, self.latest)
        while True:
            input_data = self.load_data()

            y = self.sess.run(pred, feed_dict={self.x: input_data})
            y = np.reshape(y, -1)
            y[y >= 0.3] = 1
            y[y <= 0.3] = 0
            print("the prediction is:")
            print(y[3:-3])


if __name__ == '__main__':
    unet = Unet()
    unet.predict()
