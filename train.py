# coding=utf8

import tensorflow as tf
import random
import numpy as np
import time

from model_data.model import u_net, eva_loss
from utils.utils import read_image


class UnetTrainer:
    def __init__(self):
        self.image_path = './data/train.txt'
        self.batch_size = 16

        self.image_size = [80, 416]
        # self.image_size = [80, 320]
        self.label_size = 26
        # self.label_size = 20
        self.epoch_num = 50
        self.learning_rate = 0.001

        self.val_split = 0.1

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.num = 0  # the nth of line
        # load data
        with open(self.image_path) as f:
            self.lines = f.readlines()
            split_num = int(self.val_split * len(self.lines))
            self.train_set = self.lines[split_num:]
            self.val_set = self.lines[:split_num]

        self.train()

    def load_data(self, train_set):
        """generate a batch data"""
        img_set = []
        label_set = []

        for i in range(self.batch_size):
            if self.num + self.batch_size >= len(train_set):
                pdata = train_set[len(train_set) - 1 - i]
                random.shuffle(train_set)
            else:
                pdata = train_set[self.num + i]
            img, label = pdata.split()
            label = "000" + label + "000"
            img_data = read_image("./data/picture/" + img + ".jpg")
            # img_set.append("./data/picture/" + img + ".jpg")
            img_set.append(img_data)
            label_data = list(map(eval, list(label)))
            # label_data = [10 * i for i in label_data]
            label_set.append(label_data)
        self.num += self.batch_size
        img_set = np.stack(img_set, 0)
        return img_set, label_set

    def train(self):
        """train the net"""

        # x = tf.placeholder(tf.float32, [self.batch_size, 80, 416, 3])
        x = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        # path = tf.placeholder(tf.string)
        label = tf.placeholder(tf.float32, [self.batch_size, self.label_size])
        # global_step = tf.placeholder(tf.int32, shape=[1])

        # predict
        pred = u_net(x)
        loss = eva_loss(label, pred)

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        op = opt.minimize(loss)

        # initialize
        init = tf.global_variables_initializer()
        self.sess.run(init)

        batch_train = int(len(self.train_set) / self.batch_size)
        batch_val = int(len(self.val_set) / self.batch_size)

        saver = tf.train.Saver()

        date = time.strftime("%m-%d-%H-%M")
        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        train_loss_summary = tf.summary.scalar("train_loss", loss)
        val_loss_summary = tf.summary.scalar("val_loss", loss)
        image_summary = tf.summary.image("image", x[0:2])
        mid_tensor = [[label[0, :], pred[0, :]], [label[1, :], pred[1, :]]]
        mid_tensor = tf.reshape(mid_tensor, (-1, self.label_size))
        label_summary = tf.summary.text("lable_VS_pred", tf.as_string(mid_tensor, 3))
        # pred_summary = tf.summary.text("pred", tf.as_string(pred[0:2, :], 3))

        print("train on {0} samples, val on {1} samples".format(len(self.train_set), len(self.val_set)))

        for i in range(self.epoch_num):
            self.num = 0

            # train
            for j in range(batch_train):
                t0 = time.time()
                train_set, train_label = self.load_data(train_set=self.train_set)
                # label_set = tf.Variable(label_set,False)
                # train_set = read_image(train_set)
                pred_, loss_, _, train_loss_summ = self.sess.run(
                    [pred, loss, op, train_loss_summary],
                    feed_dict={x: train_set, label: train_label})
                t1 = time.time()
                t = t1 - t0
                print(
                    "Epoch {0} Batch {1}/{2} {4:.2f}s/step train_loss {3:.4f} eta {5:.1f}s"
                        .format(i + 1, j + 1, batch_train, loss_, t, t * (batch_train - j)),
                    end='\n')
                # print("label:{0}  pred:{1}  ".format(train_label[0], pred_[0]), end='\n')
            writer.add_summary(train_loss_summ, i)

            # valid
            val_sum_loss = 0
            t0 = time.time()
            for j in range(batch_val):
                val_set, val_label = self.load_data(train_set=self.val_set)
                # val_set = tf_read_image(val_set, sess=self.sess)
                val_loss, val_loss_summ, image_summ, label_summ, pred_ = self.sess.run(
                    [loss, val_loss_summary, image_summary, label_summary, pred],
                    feed_dict={x: val_set, label: val_label})
                val_sum_loss += val_loss
            t1 = time.time()
            t = t1 - t0
            print("Epoch {0} Batch {1}/{2} {3:.2f}s/step train_loss {4:.4f} val_loss:{5:.4f}  "
                  .format(i + 1, batch_train, batch_train, t / batch_val, loss_, val_sum_loss / batch_val), end='\r')
            # print("label:{0}  pred:{1}  ".format(val_label[0], pred_[0]), end='\n')
            writer.add_summary(val_loss_summ, i)
            writer.add_summary(image_summ, i)
            writer.add_summary(label_summ, i)
            # writer.add_summary(pred_summ, i)

            if i % 5 == 4:  # save model every 3 epochs
                saver.save(self.sess,
                           "./model/epoch{2}_train_loss{0:.3f}_val_loss{1:.3f}".format(loss_,
                                                                                       val_sum_loss / batch_val, i))


if __name__ == '__main__':
    UnetTrainer()
