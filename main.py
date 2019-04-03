#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import os
import matplotlib.pyplot as plt
import subprocess as sp
from datetime import datetime
from tqdm import tqdm
import cv2
import numpy as np
import shutil
import scipy
import time

import argparse

from model import *

parser = argparse.ArgumentParser(description='process video')
parser.add_argument('-E', type=int, required=True, dest='epochs',
                    help='epochs')
parser.add_argument('-L', type=float, required=True, dest='learning',
                    help='dropout rate')
parser.add_argument('-B', type=int, required=True, dest='bsize',
                    help='batch size')
parser.add_argument('-D', type=float, required=True, dest='dropout',
                    help='dropout rate')
parser.add_argument('-M', type=str, required=True, dest='rdir',
                    help='directory for results')
parser.add_argument('-N', type=str, required=True, dest='notes',
                    help='notes')
parser.add_argument('--validate', type=bool, dest='validate',
                    help='validate model with tests')


args = parser.parse_args()

EPOCHS = args.epochs
LEARNING_RATE = args.learning
BATCH_SIZE = args.bsize
DROPOUT_RATE = args.dropout
RESULTS_DIR = args.rdir
NOTES = args.notes
VALIDATE_MODEL = args.validate

if not os.path.exists(RESULTS_DIR):
    print('results directory {} does not exist'.format(RESULTS_DIR))
    sys.exit(1)

if not os.path.isdir(RESULTS_DIR):
    print('results directory {} is not a directory'.format(RESULTS_DIR))
    sys.exit(1)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    
def save_results():
    time_str = datetime.now().strftime("%d_%m_%Y__%H_%M")
    out_dir = os.path.join(RESULTS_DIR, time_str)
    os.mkdir(out_dir)
    file_out = os.path.join(out_dir, time_str + '.txt')
    plot_fn = os.path.join(out_dir, 'loss.png')
    
    FPS = 4
    video_avi_fn = os.path.join(out_dir, 'inference.avi')
    video_mp4_fn = os.path.join(out_dir, 'inference.mp4')

    
    with open(file_out, 'w') as f:
        f.write("epochs={}, learning_rate={}, batch_size={}, dropout_rate={}\n".format(
                 EPOCHS, LEARNING_RATE, BATCH_SIZE, DROPOUT_RATE))
        f.write('notes:' + NOTES + '\n')
        
    def update(epoch, bs, loss):
        with open(file_out, 'a') as f:
            f.write('epoch={}\n'.format(epoch))
            f.write('batch_sizes={}\n'.format(bs))
            f.write('loss={}\n'.format(loss))
    

    def save_plot():
        with open(file_out, 'r') as f:
            header = f.readline()
            data = f.read()
        data = data.split('epoch=')[1:]
        epoch_loss = []
        for d in data:
            bs_idx_start = d.find('batch_sizes=[') + len('batch_sizes=[')
            bs_idx_end = d.find(']\nloss')
            l_idx_start = d.find('loss=[') + len('loss=[')
            l_idx_end = d.find(']', l_idx_start)
            bs = d[bs_idx_start: bs_idx_end]
            bs = list(map(int, bs.split(',')))
            l = d[l_idx_start: l_idx_end]
            l = list(map(float, l.split(',')))
            assert len(bs) == len(l)
            total_images = sum(bs)
            total_loss = sum(l)
            average_loss = total_loss / total_images
            epoch_loss.append(average_loss)
        
        plt.plot(epoch_loss, label='training')
        plt.xlabel('epoch')
        plt.ylabel('mean cross entropy loss')
        plt.title(header)
        plt.legend()
        plt.savefig(plot_fn)
        
    def save_video():
        test_out_dirs = os.listdir('runs/')
        test_out_dir = os.path.join('runs', test_out_dirs[-1])
        images_fn = os.listdir(test_out_dir)
        images_fn = sorted(images_fn)
        
        def process(im, txt):
            height, width, channel = im.shape
            text_height = 40
            img = np.zeros((height + text_height,width,channel), np.uint8)
            img[0:height] = im[:]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, txt,(int(width/2) - int(width/5),height + text_height - 10), font, 1,(255,0,0),2,cv2.LINE_AA)
            return img
        
        im = cv2.imread(os.path.join(test_out_dir, images_fn[0]))
        im = process(im, 'X')
        height, width, channel = im.shape

        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        video = cv2.VideoWriter(video_avi_fn, fourcc, float(FPS), (width, height))

        for im_fn in images_fn:
            frame = cv2.imread(os.path.join(test_out_dir, im_fn))
            video.write(process(frame, im_fn))
        video.release()
        
        sp.Popen(['ffmpeg','-y', '-i', video_avi_fn, video_mp4_fn])
        
        for d in test_out_dirs:
            shutil.rmtree(os.path.join('runs', d))
    
    return update, save_plot, save_video, out_dir


update, save_plot, save_video, out_dir = save_results()
MODEL_PATH = os.path.join(out_dir, 'model.ckpt')


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """                                                                                                                                                                 
    Train neural network and print out the loss during training.                                                                                                        
    :param sess: TF Session                                                                                                                                             
    :param epochs: Number of epochs                                                                                                                                     
    :param batch_size: Batch size                                                                                                                                       
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)                                                             
    :param train_op: TF Operation to train the neural network                                                                                                           
    :param cross_entropy_loss: TF Tensor for the amount of loss                                                                                                         
    :param input_image: TF Placeholder for input images                                                                                                                 
    :param correct_label: TF Placeholder for label images                                                                                                               
    :param keep_prob: TF Placeholder for dropout keep probability                                                                                                       
    :param learning_rate: TF Placeholder for learning rate                                                                                                              
    """
    
    all_size = []
    all_loss = []
    
    for epoch in range(epochs):
        epoch_batch_size = []
        epoch_loss = []
        t_start = time.time()
        
        for image, label in get_batches_fn(batch_size):
            # create feed dict: input image, correct label, keep prob, learning rate
            # loss = session.run
            feed_dict = {input_image: image,
                         correct_label: label,
                         keep_prob: DROPOUT_RATE,
                         learning_rate: LEARNING_RATE}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            
            epoch_batch_size.append(len(image))
            epoch_loss.append(loss)
            
        all_loss.append(epoch_batch_size)
        all_loss.append(epoch_loss)
        print('epoch {}/{} | {} images | loss {} | {} s'.format(epoch+1, epochs, sum(epoch_batch_size), sum(epoch_loss) / sum(epoch_batch_size), int(time.time() - t_start)))
        update(epoch, epoch_batch_size, epoch_loss)
tests.test_train_nn(train_nn)


def run():
        
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images                                                                                                       
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model                                                                                                                                     
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.                                                                             
    # You'll need a GPU with at least 10 teraFLOPS to train on.                                                                                                         
    #  https://www.cityscapes-dataset.com/
    
    with tf.Session() as sess:
        print('setup...')
        label_layer = tf.placeholder(tf.int32, (None, None, None, num_classes), name='gt_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        # Path to vgg model                                                                                                                                             
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches                                                                                                                                
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results                                                                                                                   
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network                                                        

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        layer_output = layers(layer3, layer4, layer7, num_classes)

        # TODO: Train NN using the train_nn function
        
        logits, train_op, cross_entropy_loss = optimize(layer_output, label_layer, learning_rate, num_classes )
        
        print('training...')
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        
        
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, label_layer, keep_prob, learning_rate)
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, MODEL_PATH)

        print('inference on test images...')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)                                                         
        
        print('saving plot...')
        save_plot()

        print('saving video...')
        save_video()

        # OPTIONAL: Apply the trained model to a video

if VALIDATE_MODEL:
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)

run()