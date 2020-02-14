'''
symmetrical-synthesis
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
import os
import sys
import tensorflow as tf
import numpy as np
from nets import googlenet
from tensorflow.contrib import slim

FLAGS = tf.app.flags.FLAGS

class model_builder:
    def __init__(self, batch_size=64, n_classes=1, dim_features=256, is_training=True):
        self.is_training = is_training
        self.n_classes = n_classes
        self.dim_features = dim_features
        self.batch_size = batch_size
        print('\n\n\ndim_features = {}\n\n\n'.format(self.dim_features))    
        self.batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': self.is_training
        }
    
    def vgg_preprocess(self, images, means=[123.69, 116.78, 103.94]):
        num_channels = images.get_shape().as_list()[-1]
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)
    
    def extract_cnn_features(self, inputs):
        with tf.variable_scope('googlenet'):
            # input_size should be 227
            # dim_features = 512
            google_net_model = googlenet.GoogleNet_Model(FLAGS.pretrained_model_path)
            cnn_features = google_net_model.forward(inputs)
    
        print('cnn_features', cnn_features)
        return cnn_features

    def extract_features(self, cnn_features):
        s_att = None
        with tf.variable_scope('feature_extractor'):
            features = tf.reduce_mean(cnn_features, [1, 2], keepdims=True)
            pooled_features = features
            features = slim.conv2d(features, self.dim_features, 1,
                       activation_fn=None, normalizer_fn=None)
            features = tf.squeeze(features, [1, 2])
        return features, s_att, pooled_features

    def build_features_extractor(self, input_anchor, input_pos):
        # 0. concat input_anchor and input_post
        input_images = tf.concat([input_anchor, input_pos], axis=0)
        
        input_images = self.vgg_preprocess(input_images)
       
        # 1. extract spatial features from back-bone networks
        cnn_features = self.extract_cnn_features(input_images) # 7x7x2048

        print('\n\nsimple feature extract\n\n')
        features, s_att, pooled_features = self.extract_features(cnn_features) # w/o act function!
        
        s_att = None
        logits = None

        anchor_features, pos_features = tf.split(features, 2, axis=0)
        
        return anchor_features, pos_features, logits, s_att, cnn_features

    def build_features_extractor_test(self, input_images):

        #batch_size = input_images.get_shape().as_list()[0]
        
        input_images = self.vgg_preprocess(input_images)

        cnn_features = self.extract_cnn_features(input_images)
        #print(cnn_features.get_shape().as_list())
        features, s_att, pooled_features = self.extract_features(cnn_features)
        l2_normalized_features = tf.nn.l2_normalize(features, axis=-1)
    
        return l2_normalized_features, cnn_features

## TEST
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '' # cpu mode

    batch_size, h, w, c = (64, 224, 224, 3)
    input_images = tf.placeholder(tf.float32, shape=[batch_size, h, w, c], name='input_images')

    model_builder = model_builder()
    cnn_features = model_builder.build_features_extractor_test(input_images)
    print(cnn_features)



