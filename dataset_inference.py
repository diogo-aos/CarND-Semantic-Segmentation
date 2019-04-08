import os
import subprocess as sp
import shutil
import argparse
from datetime import datetime
import glob

import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import scipy

from model import *

parser = argparse.ArgumentParser(description='process video')
parser.add_argument('model', type=str,
                    help='an integer for the accumulator')

args = parser.parse_args()



num_classes = 2
image_shape = (160, 576)  # KITTI dataset uses 160x576 images                                                                                                       
data_dir = './data'
runs_dir = './runs'

vgg_path = os.path.join(data_dir, 'vgg')

# reset graph
tf.reset_default_graph()

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # load model
    input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
    layer_output = layers(layer3, layer4, layer7, num_classes)
    logits = tf.reshape(layer_output, (-1, num_classes))

    # restore variables
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    print('inference on test images...')
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)                                                        
