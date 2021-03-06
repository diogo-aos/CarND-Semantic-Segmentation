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
parser.add_argument('model_dir', type=str,
                    help='an integer for the accumulator')
parser.add_argument('videos', type=str, nargs='+',
                    help='videos to be processed')

args = parser.parse_args()

# create model
if not os.path.exists(args.model_dir):
    print('model path does not exist')
    sys.exit(1)

if not os.path.isdir(args.model_dir):
    print('model path is not directory')
    sys.exit(1)

videos = []
for p in args.videos:
    videos.extend(glob.glob(p))
    
if not videos:
    print('no valid video paths')
    sys.exit(1)


def inference_on_image(sess, logits, keep_prob, image_input, image, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    image = scipy.misc.imresize(image, image_shape)

    # Run inference
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_input: [image]})
    # Splice out second column (road), reshape output back to image_shape
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    # If road softmax > 0.5, prediction is road
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    # Create mask based on segmentation to apply to original image
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


def info_video(video_in_fn):
    cap = cv2.VideoCapture(video_in_fn)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n_frames, fps


def load_video(video_in_fn):
    cap = cv2.VideoCapture(video_in_fn)
    # Check if camera opened successfully
    if (not cap.isOpened()):
        raise Exception("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
        yield frame

    # When everything done, release the video capture object
    cap.release()


def process_video(video_fn, model_fn, video_out_fn):
    video_out_fn_avi = video_out_fn + '.avi'
    video_out_fn_mp4 = video_out_fn + '.mp4'

    n_frames, fps = info_video(video_fn)
    image_shape = (160, 576)
    num_classes = 2
    height, width = image_shape

    # reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:
        # load model
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        layer_output = layers(layer3, layer4, layer7, num_classes)
        logits = tf.reshape(layer_output, (-1, num_classes))

        # restore variables
        saver = tf.train.Saver()
        saver.restore(sess, model_fn)

        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        video_out = cv2.VideoWriter(video_out_fn_avi, fourcc, float(fps), (width, height))

        video_in = load_video(video_fn)
        pbar = tqdm(total=n_frames)
        for frame in video_in:
            inf_frame = inference_on_image(sess, logits, keep_prob, input_image, frame, image_shape)
            video_out.write(inf_frame)
            pbar.update(1)

        video_out.release()

    p = sp.Popen(['ffmpeg', '-y', '-i', video_out_fn_avi, video_out_fn_mp4])
    p.wait()


fls = os.listdir(args.model_dir)
models = [p.rstrip('.index') for p in fls if p.endswith('.ckpt.index')]


data_dir = './data'
vgg_path = os.path.join(data_dir, 'vgg')

for model in models:
    model_fn = os.path.join(args.model_dir, model)
    for v_fn in videos:
        video_name = v_fn.split('/')[-1]
        video_name = video_name.split('.')[0]
        video_out_fn = os.path.join(args.model_dir, 'processed_{}_{}'.format(model.rstrip('.ckpt'), video_name))

        process_video(v_fn, model_fn, video_out_fn)
