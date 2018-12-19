# coding=utf8

import tensorflow as tf
import random

from model_data.model import u_net, eva_loss
from utils.utils import tf_read_image


class UNET:
    def __init__(self):
        self.image_path = './data/train.txt'
        self.batch_size = 16

        self.image_size = [80, 320]
        self.label_size = 20
        self.epoch_num = 50
        self.learning_rate = 0.001

        self.val_split = 0.1

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.num = 0  # the nth of line
        # load data
        with open(self.image_path) as f:
            self.lines = f.readlines()[:300]
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
            img_set.append("./data/picture/" + img + ".jpg")
            label_set.append(list(map(eval, list(label))))
        self.num += self.batch_size
        return img_set, label_set

    def train(self):
        """train the net"""
        x = tf.placeholder(tf.float32, [self.batch_size, 80, 320, 3])
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

        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        train_loss_summary = tf.summary.scalar("train_loss", loss)
        val_loss_summary = tf.summary.scalar("val_loss", loss)
        image_summary = tf.summary.image("image", x[0:2])
        mid_tensor = [[label[0, :], pred[0, :]], [label[1, :], pred[1, :]]]
        mid_tensor = tf.reshape(mid_tensor, (-1, 20))
        label_summary = tf.summary.text("lableVSpred", tf.as_string(mid_tensor, 3))
        # pred_summary = tf.summary.text("pred", tf.as_string(pred[0:2, :], 3))

        print("train on {0} samples, val on {1} samples".format(len(self.train_set), len(self.val_set)))

        for i in range(self.epoch_num):
            self.num = 0
            print("Epoch {0}:".format(i))

            # train
            for j in range(batch_train):
                train_set, train_label = self.load_data(train_set=self.train_set)
                # label_set = tf.Variable(label_set,False)
                train_set = tf_read_image(train_set, sess=self.sess)
                pred_, loss_, _, train_loss_summ = self.sess.run(
                    [pred, loss, op, train_loss_summary],
                    feed_dict={x: train_set, label: train_label})
                print("Batch  {0}/{1}: train_loss:{2:.8f}   ".format(j + 1, batch_train, loss_, pred_), end='\n')
                # print("label:{0}  pred:{1}  ".format(train_label[0], pred_[0]), end='\n')
            writer.add_summary(train_loss_summ, i)

            # valid
            val_sum_loss = 0
            for j in range(batch_val):
                val_set, val_label = self.load_data(train_set=self.val_set)
                val_set = tf_read_image(val_set, sess=self.sess)
                val_loss, val_loss_summ, image_summ, label_summ, pred_ = self.sess.run(
                    [loss, val_loss_summary, image_summary, label_summary, pred],
                    feed_dict={x: val_set, label: val_label})
                val_sum_loss += val_loss
            print("Batch  {0}/{1}: train_loss:{2:.8f}  val_loss:{3:.8f}  ".format(batch_train, batch_train, loss_,
                                                                                  val_sum_loss / batch_val), end='\n')
            # print("label:{0}  pred:{1}  ".format(val_label[0], pred_[0]), end='\n')
            writer.add_summary(val_loss_summ, i)
            writer.add_summary(image_summ, i)
            writer.add_summary(label_summ, i)
            # writer.add_summary(pred_summ, i)

            if i % 3 == 2:  # save model every 3 epochs
                saver.save(self.sess,
                           "./model/epoch{2}_train_loss{0:.3f}_val_loss{1:.3f}".format(loss_,
                                                                                       val_sum_loss / batch_val, i))


if __name__ == '__main__':
    UNET()
