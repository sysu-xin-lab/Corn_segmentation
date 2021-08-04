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
ptcloud_data_dir = os.path.join(root_folder, './data')
ptcloud_data_filename = os.path.join(ptcloud_data_dir, './traindata_nor.txt')
ptcloud = np.loadtxt(ptcloud_data_filename, dtype=float)
corn_labels = ptcloud[:, -1]
part_labels = ptcloud[:, -2]
ptcloud_org_corn = []
for curr_label in np.unique(corn_labels):
    curr_corn = ptcloud[corn_labels == curr_label, :]
    ptcloud_org_corn.append(curr_corn)

# Prepare inputs

parser = argparse.ArgumentParser()
parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
parser.add_argument('--save_folder',
                    '-s',
                    help='Path to folder for saving check points and summary',
                    default=os.path.join(root_folder, './save'))
parser.add_argument('--model', '-m', help='Model to use', default='pointcnn_seg')
parser.add_argument('--setting', '-x', help='Setting to use', default='myoptions_x8_2048_fps')
parser.add_argument('--epochs', help='Number of training epochs', type=int, default=5000)
parser.add_argument('--batch_size', help='Batch size ', type=int, default=8)
parser.add_argument('--log', help='Log to FILE in save folder', metavar='FILE', default='log.txt')
parser.add_argument('--no_timestamp_folder',
                    help='Dont save to timestamp folder',
                    action='store_true',
                    default=True)
parser.add_argument('--no_code_backup', help='Dont backup code', action='store_true')
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

num_epochs = args.epochs
batch_size = args.batch_size
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


def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN,
        smpw: BxN
    """
    '''
    loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('batch_loss', loss)
    tf.add_to_collection('losses', loss)
    '''
    batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    predLabel = tf.argmax(pred, 2)
    ind01 = tf.where(tf.logical_and(tf.equal(label, 0), tf.equal(predLabel, 1)))
    ind00 = tf.where(tf.logical_and(tf.equal(label, 0), tf.equal(predLabel, 0)))
    ind10 = tf.where(tf.logical_and(tf.equal(label, 1), tf.equal(predLabel, 0)))
    ind11 = tf.where(tf.logical_and(tf.equal(label, 1), tf.equal(predLabel, 1)))
    '''
    ind1 = tf.where(tf.equal(label, 1))
    loss00 = tf.gather_nd(batch_loss, ind00)
    loss01 = tf.gather_nd(batch_loss, ind01) * 2
    loss1 = tf.gather_nd(batch_loss, ind1) * 5

    loss = tf.reduce_mean(tf.concat([loss01, loss1], axis=-1))
    '''
    loss00 = tf.gather_nd(batch_loss, ind00)
    loss01 = tf.gather_nd(batch_loss, ind01) * 2
    loss10 = tf.gather_nd(batch_loss, ind10) * 8
    loss11 = tf.gather_nd(batch_loss, ind11) * 4
    loss = tf.reduce_mean(tf.concat([loss00, loss01, loss10, loss11], axis=-1))

    return loss


def rotate_ptcloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = batch_data
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, :3]
        rotated_data[k, :, :3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def genTraindata():
    train_data = np.zeros([80, NUM_POINT, NUM_DIM])
    train_label = np.zeros([80, NUM_POINT])
    train_smpw = np.zeros([80, NUM_POINT])
    val_data = np.zeros([20, NUM_POINT, NUM_DIM])
    val_label = np.zeros([20, NUM_POINT])
    val_smpw = np.ones([20, NUM_POINT])
    '''
    if split=='train':
        labelweights = np.zeros(21)
        for seg in self.semantic_labels_list:
            tmp,_ = np.histogram(seg,range(22))
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights/np.sum(labelweights)
        self.labelweights = 1/np.log(1.2+labelweights)
    elif split=='test':
        self.labelweights = np.ones(21)

    train_weights = np.zeros(2)
    train_weights[1] = np.count_nonzero(part_labels) / part_labels.shape[0]
    train_weights[0] = 1.0 - train_weights[1]
    test_weights = np.ones(2)
    '''
    for i in range(len(ptcloud_org_corn)):
        curr_corn = ptcloud_org_corn[i]
        idx = np.random.choice(np.arange(len(curr_corn)), size=NUM_POINT)
        if i < 80:
            train_data[i, ...] = curr_corn[idx, :NUM_DIM]
            train_label[i, ...] = curr_corn[idx, -2]
            for j in range(len(idx)):
                train_smpw[i, j] = train_weights[int(curr_corn[idx[j], -2])]
        else:
            val_data[i - 80, ...] = curr_corn[idx, :NUM_DIM]
            val_label[i - 80, ...] = curr_corn[idx, -2]
    # print(train_label)
    # print(train_smpw)
    idx = np.arange(80)
    np.random.shuffle(idx)
    train_data = train_data[idx, ...]
    train_data = rotate_ptcloud(train_data)
    train_label = train_label[idx, ...]
    return train_data, train_label, train_smpw, val_data, val_label, val_smpw


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(None, NUM_POINT, setting.data_dim), name='pts_fts')
    labels_seg = tf.placeholder(tf.int64, shape=(None, NUM_POINT), name='labels_seg')
    labels_weights = tf.placeholder(tf.float32, shape=(None, NUM_POINT), name='labels_weights')

    ######################################################################
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    features_augmented = None
    if setting.data_dim > 3:
        log_string('Only 3D normals are supported!')
        exit()
    else:
        points_sampled = pts_fts_sampled
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)

    labels_sampled = tf.gather_nd(labels_seg, indices=indices, name='labels_sampled')
    labels_weights_sampled = tf.gather_nd(labels_weights,
                                          indices=indices,
                                          name='labels_weight_sampled')

    net = model.Net(points_augmented, features_augmented, is_training, setting)
    logits = net.logits
    probs = tf.nn.softmax(logits, name='probs')
    predictions = tf.argmax(probs, axis=-1, name='predictions')
    '''
    losses = tf.losses.sparse_softmax_cross_entropy(labels=labels_sampled,
                                                    logits=logits,
                                                    weights=labels_weights_sampled)
    '''
    losses = get_loss(logits, labels_sampled, labels_weights_sampled)
    tf.summary.scalar('losses', losses)
    correct = tf.equal(tf.argmax(logits, 2), tf.to_int64(labels_sampled))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batch_size)
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('metrics'):
        loss_mean_op, loss_mean_update_op = tf.metrics.mean(losses)
        t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(labels_sampled,
                                                            predictions,
                                                            weights=labels_weights_sampled)
        t_1_per_class_acc_op, t_1_per_class_acc_update_op = \
            tf.metrics.mean_per_class_accuracy(labels_sampled, predictions, NUM_CLASS,
                                               weights=labels_weights_sampled)
    reset_metrics_op = tf.variables_initializer(
        [var for var in tf.local_variables() if var.name.split('/')[0] == 'metrics'])

    tf.summary.scalar('loss/train', tensor=loss_mean_op, collections=['train'])
    tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])
    tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

    tf.summary.scalar('loss/val', tensor=loss_mean_op, collections=['val'])
    tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_op, collections=['val'])
    tf.summary.scalar('t_1_per_class_acc/val', tensor=t_1_per_class_acc_op, collections=['val'])

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base,
                                           global_step,
                                           setting.decay_steps,
                                           setting.decay_rate,
                                           staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])
    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op,
                                               momentum=setting.momentum,
                                               use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(losses + reg_loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    # backup all code
    if not args.no_code_backup:
        shutil.copy(os.path.join(root_folder, 'train_val_seg.py'), save_folder)
        shutil.copy(os.path.join(root_folder, 'pointcnn_seg.py'), save_folder)
        shutil.copy(os.path.join(root_folder, 'pointcnn.py'), save_folder)
        shutil.copy(os.path.join(root_folder, 'pointcnn_seg/myoptions_x8_2048_fps.py'), save_folder)
        log_string('copy files to %s!' % save_folder)

    folder_ckpt = os.path.join(save_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(save_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    log_string('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_val_op = tf.summary.merge_all('val')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(init_op)

        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            log_string('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))
        ops = {
            'reset_metrics_op': reset_metrics_op,
            'train_op': train_op,
            'loss_mean_update_op': loss_mean_update_op,
            't_1_acc_update_op': t_1_acc_update_op,
            't_1_per_class_acc_update_op': t_1_per_class_acc_update_op,
            'loss_mean_op': loss_mean_op,
            't_1_acc_op': t_1_acc_op,
            't_1_per_class_acc_op': t_1_per_class_acc_op,
            'summaries_op': summaries_op,
            'global_step': global_step,
            'pts_fts': pts_fts,
            'indices': indices,
            'xforms': xforms,
            'rotations': rotations,
            'jitter_range': jitter_range,
            'labels_seg': labels_seg,
            'labels_weights': labels_weights,
            'summaries_val_op': summaries_val_op,
            'is_training': is_training,
            'pred': predictions
        }

        for epoch in range(num_epochs):
            log_string("=======Epoch: %.03d========" % (epoch + 1))
            train_data, train_label, train_smpw, val_data, val_label, val_smpw = genTraindata()
            train_one_epoch(sess, train_data, train_label, train_smpw, ops, summary_writer)
            curr_val_data, curr_val_label, pred_label, output_indices, curr_step = eval_one_epoch(
                sess, val_data, val_label, val_smpw, ops, summary_writer)
            if (epoch + 1) % 100 == 0:
                save_path = saver.save(
                    sess, os.path.join(save_folder, 'epoch_' + str(curr_step) + '.ckpt'))
                sp.io.savemat(
                    os.path.join(save_folder, 'epoch_' + str(curr_step) + '.mat'), {
                        'val_data': curr_val_data,
                        'val_label': curr_val_label,
                        'pred_label': pred_label,
                        'index': output_indices
                    })
                log_string("Model saved in file: %s" % save_path)


def eval_one_epoch(sess, test_data, test_label, test_smpw, ops, summary_writer):

    sess.run(ops['reset_metrics_op'])
    num_batches = test_data.shape[0] // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        cur_batch_data = test_data[start_idx:end_idx, ...]
        cur_batch_label = test_label[start_idx:end_idx, ...]
        cur_batch_smpw = test_smpw[start_idx:end_idx, ...]
        xforms_np, rotations_np = pf.get_xforms(batch_size,
                                                rotation_range=rotation_range_val,
                                                scaling_range=scaling_range_val,
                                                order=setting.rotation_order)
        pred_val, output_indices, _, _, _ = sess.run(
            [
                ops['pred'], ops['indices'], ops['loss_mean_update_op'], ops['t_1_acc_update_op'],
                ops['t_1_per_class_acc_update_op']
            ],
            feed_dict={
                ops['pts_fts']: cur_batch_data,
                ops['indices']: pf.get_indices(batch_size, sample_num, NUM_POINT),
                ops['xforms']: xforms_np,
                ops['rotations']: rotations_np,
                ops['jitter_range']: np.array([jitter]),
                ops['labels_seg']: cur_batch_label,
                ops['labels_weights']: cur_batch_smpw,
                ops['is_training']: False
            })
    loss_val, t_1_acc_val, t_1_per_class_acc_val, summaries_val, step = sess.run([
        ops['loss_mean_op'], ops['t_1_acc_op'], ops['t_1_per_class_acc_op'],
        ops['summaries_val_op'], ops['global_step']
    ])
    summary_writer.add_summary(summaries_val, step)
    log_string('{}-[Val  ]-Average:      Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'.format(
        datetime.now(), loss_val, t_1_acc_val, t_1_per_class_acc_val))

    return cur_batch_data, cur_batch_label, pred_val, output_indices, step


def train_one_epoch(sess, train_data, train_label, train_smpw, ops, summary_writer):

    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    sess.run(ops['reset_metrics_op'])
    ######################################################################
    # Training
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        cur_batch_data = train_data[start_idx:end_idx, ...]
        cur_batch_label = train_label[start_idx:end_idx, ...]
        cur_batch_smpw = train_smpw[start_idx:end_idx, ...]

        offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
        offset = max(offset, -sample_num * setting.sample_num_clip)
        offset = min(offset, sample_num * setting.sample_num_clip)
        sample_num_train = sample_num + offset
        xforms_np, rotations_np = pf.get_xforms(batch_size,
                                                rotation_range=rotation_range,
                                                scaling_range=scaling_range,
                                                order=setting.rotation_order)
        sess.run(
            [
                ops['train_op'], ops['loss_mean_update_op'], ops['t_1_acc_update_op'],
                ops['t_1_per_class_acc_update_op']
            ],
            feed_dict={
                ops['pts_fts']: cur_batch_data,
                ops['indices']: pf.get_indices(batch_size, sample_num_train, NUM_POINT),
                ops['xforms']: xforms_np,
                ops['rotations']: rotations_np,
                ops['jitter_range']: np.array([jitter]),
                ops['labels_seg']: cur_batch_label,
                ops['labels_weights']: cur_batch_smpw,
                ops['is_training']: True
            })
        if (batch_idx + 1) == num_batches:
            loss, t_1_acc, t_1_per_class_acc, summaries, step = sess.run([
                ops['loss_mean_op'], ops['t_1_acc_op'], ops['t_1_per_class_acc_op'],
                ops['summaries_op'], ops['global_step']
            ])
            summary_writer.add_summary(summaries, step)
            log_string(
                '{}-[Train]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'.format(
                    datetime.now(), step, loss, t_1_acc, t_1_per_class_acc))

        ######################################################################


if __name__ == '__main__':
    train()
    LOG_FOUT.close()
