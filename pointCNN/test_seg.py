#!/usr/bin/python3
"""Training and Validation On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import scipy as sp
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime

# Load ALL data
root_folder = os.path.dirname(os.path.abspath(__file__))
ptcloud_data_dir = os.path.join(root_folder, '../data')
ptcloud_data_filename = os.path.join(ptcloud_data_dir, 'testdata_nor.txt')
ptcloud = np.loadtxt(ptcloud_data_filename, dtype=float)
corn_labels = ptcloud[:, -1]
ptcloud_org_corn = []
for curr_label in np.unique(corn_labels):
    curr_corn = ptcloud[corn_labels == curr_label, :]
    ptcloud_org_corn.append(curr_corn)

# Prepare inputs
parser = argparse.ArgumentParser()
parser.add_argument('--load_ckpt',
                    '-l',
                    help='Path to a check point file for load',
                    default='save/epoch_final.ckpt')
parser.add_argument('--save_folder',
                    '-s',
                    help='Path to folder for saving check points and summary',
                    default=os.path.join(root_folder, './save'))
parser.add_argument('--model', '-m', help='Model to use', default='pointcnn_seg')
parser.add_argument('--setting', '-x', help='Setting to use', default='myoptions_x8_2048_fps')
parser.add_argument('--epochs', help='Number of training epochs', type=int, default=5000)
parser.add_argument('--log', help='Log file in save folder', metavar='FILE', default='log_test.txt')
parser.add_argument('--no_timestamp_folder',
                    help='Dont save to timestamp folder',
                    action='store_true',
                    default=True)
args = parser.parse_args()

if not args.no_timestamp_folder:
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_folder = os.path.join(args.save_folder,
                               '%s_%s_%s_%d' % (args.model, args.setting, time_string, os.getpid()))
else:
    save_folder = args.save_folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

LOG_FOUT = open(os.path.join(save_folder, args.log), 'w')

model = importlib.import_module(args.model)
setting_path = os.path.join(os.path.dirname(__file__), args.model)
sys.path.append(setting_path)
setting = importlib.import_module(args.setting)

NUM_POINT = setting.sample_num
NUM_CLASS = setting.num_class
NUM_DIM = 3
BATCH_SIZE = len(ptcloud_org_corn)

sample_num = setting.sample_num
step_val = setting.step_val
label_weights_list = setting.label_weights
rotation_range = setting.rotation_range
rotation_range_val = setting.rotation_range_val
scaling_range = setting.scaling_range
scaling_range_val = setting.scaling_range_val
jitter = setting.jitter
jitter_val = setting.jitter_val
train_weights = setting.label_weights


def genTestdata():
    test_data = np.zeros([BATCH_SIZE, NUM_POINT, NUM_DIM])
    for i in range(BATCH_SIZE):
        curr_corn = ptcloud_org_corn[i]
        idx = np.random.choice(np.arange(len(curr_corn)), size=NUM_POINT)
        test_data[i, ...] = curr_corn[idx, :NUM_DIM]
    return test_data


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def test():
    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(None, NUM_POINT, setting.data_dim), name='pts_fts')

    ######################################################################
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    features_augmented = None
    if setting.data_dim > 3:
        log_string('Only 3D normals are supported!')
        exit()
    else:
        points_sampled = pts_fts_sampled
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)

    net = model.Net(points_augmented, features_augmented, is_training, setting)
    logits = net.logits
    probs = tf.nn.softmax(logits, name='probs')
    predictions = tf.argmax(probs, axis=-1, name='predictions')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            log_string('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))
        ops = {
            'global_step': global_step,
            'pts_fts': pts_fts,
            'indices': indices,
            'xforms': xforms,
            'rotations': rotations,
            'jitter_range': jitter_range,
            'is_training': is_training,
            'pred': predictions
        }

        testdata = genTestdata()
        pred_label, pred_indices = test_one_epoch(sess, testdata, ops)

        sp.io.savemat(os.path.join(ptcloud_data_dir, 'pointcnn_test.mat'), {
            'val_data': testdata,
            'index': pred_indices,
            'pred_label': pred_label
        })
        log_string("test finished! results saved to ptcloud_data_dir/pointcnn_test.mat")


def test_one_epoch(sess, test_data, ops):
    num_batches = test_data.shape[0] // BATCH_SIZE
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        cur_batch_data = test_data[start_idx:end_idx, ...]
        xforms_np, rotations_np = pf.get_xforms(BATCH_SIZE,
                                                rotation_range=rotation_range_val,
                                                scaling_range=scaling_range_val,
                                                order=setting.rotation_order)
        pred_val = sess.run(
            [ops['pred'], ops['indices']],
            feed_dict={
                ops['pts_fts']: cur_batch_data,
                ops['indices']: pf.get_indices(BATCH_SIZE, sample_num, NUM_POINT),
                ops['xforms']: xforms_np,
                ops['rotations']: rotations_np,
                ops['jitter_range']: np.array([jitter]),
                ops['is_training']: False
            })
    return pred_val


if __name__ == '__main__':
    test()
    LOG_FOUT.close()
