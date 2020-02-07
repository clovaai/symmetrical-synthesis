'''
symmetrical-synthesis
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from dataset import data_load
import models
import losses
import utils

tf.app.flags.DEFINE_string('run_gpu', '0', 'use single gpu')
tf.app.flags.DEFINE_string('save_path', '/where/your/folder', '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, 'imagenet/pretrained_model.ckpt')
tf.app.flags.DEFINE_boolean('restore_model', False, '')
tf.app.flags.DEFINE_string('image_path', '/where/your/saved/image/folder', '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_integer('num_readers', 4, '')
tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_integer('dim_features', 256, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_workers', 4, '')
tf.app.flags.DEFINE_integer('max_to_keep', 10, 'how many do you want to save models?')
tf.app.flags.DEFINE_integer('save_model_steps', 2000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_integer('max_steps', 10000, '')
tf.app.flags.DEFINE_string('losses', 'npair', 'class, npair, ...')
tf.app.flags.DEFINE_boolean('vis_sim_matrix', True, '')

# for train setup
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'define your learing strategy')
tf.app.flags.DEFINE_integer('decay_steps', 100000, '')
tf.app.flags.DEFINE_float('decay_ratio', 0.1, '')
tf.app.flags.DEFINE_integer('decay_stop_steps', None, '')
tf.app.flags.DEFINE_float('decay_stop_value', 0.00001, '')
tf.app.flags.DEFINE_boolean('adabound', False, '')
tf.app.flags.DEFINE_boolean('clip_gradient', False, '')
tf.app.flags.DEFINE_float('clip_value', 5.0, '')
tf.app.flags.DEFINE_integer('n_class', 98, '98 for CARS')

# for loss combination
tf.app.flags.DEFINE_float('w_npair', 1.0, '')
tf.app.flags.DEFINE_float('w_angular', 2.0, '')
tf.app.flags.DEFINE_float('w_symm', 1.0, '')
tf.app.flags.DEFINE_float('w_symm_angular', 1.0, '')
tf.app.flags.DEFINE_float('w_wd', 5e-3, '')
tf.app.flags.DEFINE_float('w_fc', 10.0, '')
tf.app.flags.DEFINE_float('w_reg', 1.0, '')

# for angular loss
tf.app.flags.DEFINE_float('degree', 45.0, '')
tf.app.flags.DEFINE_boolean('angular_l2norm', False, '')
tf.app.flags.DEFINE_boolean('with_npair', False, '')

# for symm_npair.
tf.app.flags.DEFINE_string('symm_type', 'sphere', 'sphere | euclidean')
tf.app.flags.DEFINE_boolean('l2_reg', True, '')

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    ######################### System setup
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.run_gpu
    utils.prepare_checkpoint_path(FLAGS.save_path, FLAGS.restore_model)
    
    ######################### Model setup
    real_batch_size = FLAGS.batch_size // 2
    
    input_anchor = tf.placeholder(tf.float32, shape=[real_batch_size, FLAGS.input_size, FLAGS.input_size, 3], name='input_anchor')
    
    input_gt = tf.placeholder(tf.float32, shape=[real_batch_size, 1], name='input_gt')
        
    input_pos = tf.placeholder(tf.float32, shape=[real_batch_size, FLAGS.input_size, FLAGS.input_size, 3], name='input_pos')
    
    ######################### Build features extractor
    model_builder = models.model_builder(batch_size=FLAGS.batch_size, n_classes=FLAGS.num_classes, dim_features=FLAGS.dim_features)
    anchor_features, pos_features, logits, s_att, cnn_features = model_builder.build_features_extractor(input_anchor, input_pos)
    
    ######################### Losses setup
    loss_builder = losses.loss_builder()
    
    loss_list = utils.loss_parser(FLAGS.losses)

    total_loss = 0.0

    if 'npair' in loss_list or 'npair_loss' in loss_list:
        npair_loss = loss_builder.npair_loss(input_gt, anchor_features, pos_features)
        tf.summary.scalar('npair_loss', npair_loss)
        total_loss += FLAGS.w_npair * npair_loss
    if 'angular' in loss_list or 'angular_loss' in loss_list:
        angular_loss = loss_builder.angular_loss(input_gt, anchor_features, pos_features, bs=FLAGS.batch_size, degree=FLAGS.degree, l2_norm=FLAGS.angular_l2norm, with_l2reg=FLAGS.l2_reg, with_npair=FLAGS.with_npair)
        tf.summary.scalar('angular_loss', angular_loss)
        total_loss += FLAGS.w_angular * angular_loss
    if 'symm_npair' in loss_list:
        # for now, doing test
        symm_npair_loss, sim_concat = loss_builder.symm_npair_loss(input_gt, anchor_features, pos_features, l2_norm=False, with_l2reg=FLAGS.l2_reg, symm_type=FLAGS.symm_type)
        tf.summary.scalar('symm_npair_loss', symm_npair_loss)
        total_loss += FLAGS.w_symm * symm_npair_loss
    if 'symm_angular' in loss_list:
        symm_angular_loss, sim_concat = loss_builder.symm_angular_loss(input_gt, anchor_features, pos_features, bs=FLAGS.batch_size, degree=FLAGS.degree, l2_norm=False, with_l2reg=FLAGS.l2_reg)
        tf.summary.scalar('symm_angular_loss', symm_angular_loss)
        total_loss += FLAGS.w_symm_angular * symm_angular_loss
    
    tf.summary.scalar('total_loss', total_loss)
    
    ######################### Training setup
    global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)

    train_vars = tf.trainable_variables()
    
    def weight_check(name):
        print(name)
        return 'BatchNorm' not in name and 'batch_normalization' not in name and 'center' not in name
    wd_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if weight_check(v.name)])
    tf.summary.scalar('wd_loss', wd_loss)
    
    total_loss = total_loss + FLAGS.w_wd * wd_loss

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # split backbone and fc layers
    #
    learning_rate = utils.configure_learning_rate(FLAGS.learning_rate, global_step)
    if FLAGS.decay_stop_steps is not None:
        learning_rate = tf.cond(global_step < FLAGS.decay_stop_steps, lambda: learning_rate, lambda: FLAGS.decay_stop_value)
    
    tf.summary.scalar('learning_rate', learning_rate)
    
    optimizer = utils.configure_optimizer(learning_rate)

    gradients = optimizer.compute_gradients(total_loss, var_list=train_vars)
    new_gradients = []
    for gradient, var in gradients:
        if FLAGS.clip_gradient:
            gradient = tf.clip_by_norm(gradient, FLAGS.clip_value)
        if 'feature_extractor' in var.name:
            gradient = tf.convert_to_tensor(gradient)
            gradient = FLAGS.w_fc * gradient
        new_gradients.append((gradient, var))
    
    grad_updates = optimizer.apply_gradients(new_gradients, global_step=global_step)
    
    with tf.control_dependencies([grad_updates] + update_ops):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.save_path, tf.get_default_graph())

    ######################### Train process
    ## resnet
    init = tf.global_variables_initializer()
    
    data_generator = data_load.get_batch(image_path=FLAGS.image_path,
                                            num_workers=FLAGS.num_workers,
                                            batch_size=FLAGS.batch_size,
                                            input_size=FLAGS.input_size)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init_iter = 0
        if FLAGS.restore_model:
            ckpt = tf.train.latest_checkpoint(FLAGS.save_path)
            saver.restore(sess, ckpt)
            init_iter = int(ckpt.split('-')[-1])
            print('%s loaded and current global step: %d' % (ckpt, init_iter))
        else:
            sess.run(init)
        
        start_time = time.time()
        
        check_time = 100
        time_array = np.zeros([check_time], dtype='float32')

        for iter_val in range(init_iter + 1, FLAGS.max_steps + 1):
            data = next(data_generator)
            loaded_anchor = np.asarray(data[0])
            loaded_pos = np.asarray(data[1])
            gts = np.asarray(data[2], dtype='float32').reshape([-1,1])
            feed_dict = {input_anchor: loaded_anchor,
                         input_pos: loaded_pos,
                         input_gt: gts}
            
            train_start = time.time()
            
            loss_val, _ = sess.run([total_loss, train_op], feed_dict=feed_dict)
 
            time_array[iter_val % check_time] = time.time() - train_start
            
            if iter_val != 0 and iter_val % FLAGS.save_summary_steps == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=iter_val)
                
                used_time = time.time() - start_time
                avg_time_per_step = used_time / FLAGS.save_summary_steps
                avg_examples_per_second = (FLAGS.save_summary_steps * FLAGS.batch_size) / used_time
                
                if loss_val == 'nan' or loss_val is np.nan:
                    exit()
                print('step %d, loss %.4f, %.2f seconds/step, %.2f examples/second'
                      % (iter_val, loss_val, avg_time_per_step, avg_examples_per_second))
                start_time = time.time()

            if iter_val != 0 and iter_val % FLAGS.save_model_steps == 0:
                checkpoint_fn = os.path.join(FLAGS.save_path, 'model.ckpt')
                saver.save(sess, checkpoint_fn, global_step=iter_val)

        print('')
        print('*' * 30)
        print(' Training done!!! ')
        print('*' * 30)
        print('')

if __name__ == '__main__':
    tf.app.run()
