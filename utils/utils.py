# coding=utf8

import tensorflow as tf


def tf_read_image(image_set, sess):
    """read image form image_set path"""
    image_data_set = []
    for i in range(len(image_set)):
        image_raw_data = tf.gfile.GFile(image_set[i], 'rb').read()
        image_data = tf.image.decode_jpeg(image_raw_data)
        image_data = tf.image.crop_to_bounding_box(image_data, 0, 78, 100, 360)
        image_data = tf.image.resize_images(image_data, [80, 320])
        # image_data = tf.expand_dims(image_data, 0)
        image_data_set.append(image_data)

    image_data_set = tf.stack(image_data_set, 0)
    return image_data_set.eval(session=sess) / 255
