'''
symmetrical-synthesis
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import itertools

tf.app.flags.DEFINE_float('symm_ratio', 2.0, 'should be more than 1.0')
tf.app.flags.DEFINE_float('symm_norm_ratio', 1.0, '')
tf.app.flags.DEFINE_boolean('symm_norm', False, '')
FLAGS = tf.app.flags.FLAGS

EPS = 1e-12

class loss_builder:
    def __init__(self):
        return
    
    def npair_loss(self, gt, anchor, pos):
        '''
        build tf.contrib.losses.metric_learning.npairs_loss
        anchor, pos: features, which are extracted from a backbone networks
        NOTE!!! features should not be l2-normalized
        '''
        gt = tf.reshape(gt, [-1])

        loss = tf.contrib.losses.metric_learning.npairs_loss(
                labels=gt,
                embeddings_anchor=anchor,
                embeddings_positive=pos,
                reg_lambda=0.002)

        return loss
    
    def tf_get_symmetric_point(self, anchor, pos):
        P = pos
        A = anchor # basis
        A_norm = tf.linalg.norm(A, axis=1, keepdims=True)
        U = A / A_norm
        R = A + tf.reduce_sum(tf.multiply((P - A), U), axis=1, keepdims=True) * U
        Q = FLAGS.symm_norm_ratio * (FLAGS.symm_ratio * R - P)
        
        if FLAGS.symm_norm:
            Q = Q / tf.linalg.norm(Q, axis=1, keepdims=True) * A_norm

        return Q
    
    def symm_npair_loss(self, gt, anchor, pos, l2_norm=False, with_l2reg=False, symm_type='sphere'):
        gt = tf.reshape(gt, [-1])

        if with_l2reg:
            reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(anchor), 1))
            reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(pos), 1))
            l2loss = tf.multiply(0.25 * 0.002, reg_anchor + reg_positive, name='l2loss_am_npair')
        else:
            l2loss = 0.0
        
        if l2_norm:
            anchor = tf.nn.l2_normalize(anchor, axis=1)
            pos = tf.nn.l2_normalize(pos, axis=1)

        ## get symm vectors
        print('\n\n\n%s type symmetric points used!!! \n\n\n' % 'sphere')
    
        # e.g., sym1 | anchor | pos
        # e.g., anchor | pos | sym2
        sym1 = self.tf_get_symmetric_point(anchor, pos)
        sym2 = self.tf_get_symmetric_point(pos, anchor)

        ## similarity matrix fTf
        
        pts_list = []
        pts_list.append(anchor)
        pts_list.append(pos)
        pts_list.append(sym1)
        pts_list.append(sym2)

        sim_mat_list = []

        selected_pts_list = itertools.combinations(pts_list, 2) # use combination! more simple, more stable.
        
        for selected_pts in selected_pts_list:
            sim_mat = tf.matmul(selected_pts[0], selected_pts[1], transpose_a=False, transpose_b=True)
            sim_mat_list.append(tf.expand_dims(sim_mat, axis=-1))
        
        sim_concat = tf.concat(sim_mat_list, axis=-1)
        
        similarity_matrix = tf.reduce_max(sim_concat, axis=-1)
        
        # do softmax cross-entropy
        lshape = tf.shape(gt)
        labels = tf.reshape(gt, [lshape[0], 1])
        
        labels_remapped = tf.to_float(tf.equal(labels, tf.transpose(labels)))
        labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

        xent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
        xent_loss = tf.reduce_mean(xent_loss, name='xentropy_am_npair')

        return l2loss + xent_loss, tf.expand_dims(tf.argmax(sim_concat, axis=-1), axis=-1)#tf.expand_dims(similarity_matrix, axis=-1)#sim_concat    

    def get_angular_sim_mat(self, anchor, pos, sq_tan_alpha, batch_size):
        xaTxp = tf.matmul(anchor, pos, transpose_a=False, transpose_b=True)
        sim_matrix_1 = tf.multiply(2.0 * (1.0 + sq_tan_alpha) * xaTxp, tf.eye(batch_size, dtype=tf.float32))
        xaPxpTxn = tf.matmul((anchor + pos), pos, transpose_a=False, transpose_b=True)
        sim_matrix_2 = tf.multiply(4.0 * sq_tan_alpha * xaPxpTxn, tf.ones_like(xaPxpTxn, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32))
        similarity_matrix = sim_matrix_1 + sim_matrix_2

        return similarity_matrix

    def angular_loss(self, gt, anchor, pos, bs, degree=45, l2_norm=False, with_l2reg=False, with_npair=False):
        gt = tf.reshape(gt, [-1])
        
        if with_l2reg:
            reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(anchor), 1))
            reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(pos), 1))
            l2loss = tf.multiply(0.25 * 0.002, reg_anchor + reg_positive, name='l2loss_angular')
        else:
            l2loss = 0.0
        
        ## l2_normalize
        if l2_norm:
            anchor = tf.nn.l2_normalize(anchor, axis=1)
            pos = tf.nn.l2_normalize(pos, axis=1)

        alpha = np.deg2rad(degree)
        sq_tan_alpha = np.tan(alpha) ** 2

        batch_size = bs // 2

        # 2(1+(tan(alpha))^2 * xaTxp)
        xaTxp = tf.matmul(anchor, pos, transpose_a=False, transpose_b=True)
        sim_matrix_1 = tf.multiply(2.0 * (1.0 + sq_tan_alpha) * xaTxp, tf.eye(batch_size, dtype=tf.float32))

        # 4((tan(alpha))^2(xa + xp)Txn
        xaPxpTxn = tf.matmul((anchor + pos), pos, transpose_a=False, transpose_b=True)
        sim_matrix_2 = tf.multiply(4.0 * sq_tan_alpha * xaPxpTxn, tf.ones_like(xaPxpTxn, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32))

        # similarity_matrix
        if with_npair:
            print('\nangular with nested npair loss\n')
            sim_matrix_3 = xaTxp
        else:
            sim_matrix_3 = 0.0
        similarity_matrix = sim_matrix_1 + sim_matrix_2 + sim_matrix_3

        # do softmax cross-entropy
        lshape = tf.shape(gt)
        labels = tf.reshape(gt, [lshape[0], 1])
        
        labels_remapped = tf.to_float(tf.equal(labels, tf.transpose(labels)))
        labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

        xent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
        xent_loss = tf.reduce_mean(xent_loss, name='xentropy_angular')

        return l2loss + xent_loss

    def symm_angular_loss(self, gt, anchor, pos, bs, degree=45, l2_norm=False, with_l2reg=False):
        gt = tf.reshape(gt, [-1])
        
        if with_l2reg:
            reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(anchor), 1))
            reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(pos), 1))
            l2loss = tf.multiply(0.25 * 0.002, reg_anchor + reg_positive, name='l2loss_angular')
        else:
            l2loss = 0.0
        
        ## l2_normalize
        if l2_norm:
            anchor = tf.nn.l2_normalize(anchor, axis=1)
            pos = tf.nn.l2_normalize(pos, axis=1)

        alpha = np.deg2rad(degree)
        sq_tan_alpha = np.tan(alpha) ** 2

        batch_size = bs // 2
 
        ## get symm vectors
        sym1 = self.tf_get_symmetric_point(anchor, pos)
        sym2 = self.tf_get_symmetric_point(pos, anchor)

        sim_org = tf.expand_dims(self.get_angular_sim_mat(anchor, pos, sq_tan_alpha, batch_size), axis=-1)
        sim_1 = tf.expand_dims(self.get_angular_sim_mat(anchor, sym1, sq_tan_alpha, batch_size), axis=-1)
        sim_2 = tf.expand_dims(self.get_angular_sim_mat(anchor, sym2, sq_tan_alpha, batch_size), axis=-1)
        sim_3 = tf.expand_dims(self.get_angular_sim_mat(pos, anchor, sq_tan_alpha, batch_size), axis=-1)
        sim_4 = tf.expand_dims(self.get_angular_sim_mat(pos, sym1, sq_tan_alpha, batch_size), axis=-1)
        sim_5 = tf.expand_dims(self.get_angular_sim_mat(pos, sym2, sq_tan_alpha, batch_size), axis=-1)
        sim_6 = tf.expand_dims(self.get_angular_sim_mat(sym1, pos, sq_tan_alpha, batch_size), axis=-1)
        sim_7 = tf.expand_dims(self.get_angular_sim_mat(sym1, anchor, sq_tan_alpha, batch_size), axis=-1)
        sim_8 = tf.expand_dims(self.get_angular_sim_mat(sym1, sym2, sq_tan_alpha, batch_size), axis=-1)
        sim_9 = tf.expand_dims(self.get_angular_sim_mat(sym2, pos, sq_tan_alpha, batch_size), axis=-1)
        sim_10 = tf.expand_dims(self.get_angular_sim_mat(sym2, anchor, sq_tan_alpha, batch_size), axis=-1)
        sim_11 = tf.expand_dims(self.get_angular_sim_mat(sym2, sym1, sq_tan_alpha, batch_size), axis=-1)
        
        sim_concat = tf.concat([sim_org, sim_1, sim_2, sim_3, sim_4, sim_5, sim_6, sim_7, sim_8, sim_9, sim_10, sim_11], axis=-1)

        similarity_matrix = tf.reduce_max(sim_concat, axis=-1)
        
        # do softmax cross-entropy
        lshape = tf.shape(gt)
        labels = tf.reshape(gt, [lshape[0], 1])
        
        labels_remapped = tf.to_float(tf.equal(labels, tf.transpose(labels)))
        labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

        xent_loss = tf.nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
        xent_loss = tf.reduce_mean(xent_loss, name='xentropy_angular')

        return l2loss + xent_loss, sim_concat
