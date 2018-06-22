#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:24
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet_instance.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet的实例分割部分
"""
import argparse
import os.path as ops

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_instance_segmentation
from lanenet_model import lanenet_cluster
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The test image path')
    parser.add_argument('--net', type=str, help='The base net model', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The weights path')

    return parser.parse_args()


def test_net(image, weights_path, net_flag):
    """

    :param image_path:
    :param weights_path:
    :param net_flag:
    :return:
    """
    if len(image.shape) < 3 or image.shape[2] == 1:
        image = np.stack([image,image,image], axis = 2)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    ori_image = image
    image = image - [103.939, 116.779, 123.68]
    image = np.expand_dims(image, axis=0)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    label_tensor = tf.zeros(shape=[1, 256, 512], dtype=tf.float32, name='label_tensor')
    phase_tensor = tf.constant('test', dtype=tf.string)

    net = lanenet_instance_segmentation.LaneNetInstanceSeg(net_flag=net_flag, phase=phase_tensor)
    net_out = net.compute_loss(input_tensor=input_tensor, label=label_tensor, name='lanenet_loss')

    out_logits = net_out['embedding']

    saver = tf.train.Saver()
    cluster = lanenet_cluster.LaneNetCluster()

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        predict_map = sess.run(out_logits, feed_dict={input_tensor: image})
    sess.close()

    return predict_map[0]
