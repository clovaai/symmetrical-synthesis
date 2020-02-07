'''
symmetrical-synthesis
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import os
import glob
from PIL import Image, ImageDraw, ImageFont

FLAGS = tf.app.flags.FLAGS

def prepare_checkpoint_path(save_path, restore):
    if not tf.gfile.Exists(save_path):
        tf.gfile.MkDir(save_path)
    else:
        if not restore:
            tf.gfile.DeleteRecursively(save_path)
            tf.gfile.MkDir(save_path)

def configure_learning_rate(learning_rate_init_value, global_step):
    learning_rate = tf.train.exponential_decay(learning_rate_init_value, global_step, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_ratio, staircase=True) # 10000
    return learning_rate

def configure_optimizer(learning_rate):
    print('\nAdam used!!!\n')
    return tf.train.AdamOptimizer(learning_rate)

def get_restore_op(vgg_path, train_vars, check=False):
    vgg_19_vars = [var for var in train_vars if var.name.startswith('vgg')]
    if check:
        print_vars(vgg_19_vars)
    variable_restore_op = slim.assign_from_checkpoint_fn(
                            vgg_path,
                            vgg_19_vars,
                            ignore_missing_vars=True)
    return variable_restore_op

def print_vars(var_list):
    print('')
    for var in var_list:
        print(var)
    print('')

def loss_parser(str_loss):
    '''
    NOTE!!! str_loss should be like 'mse,perceptual,texture,adv'...
    '''
    selected_loss_array = str_loss.split(',')
    return selected_loss_array

def rollback_parser(str_rollback):
    rollback_array = str_rollback.split(',')
    return [int(element) for element in rollback_array]

def get_last_ckpt_path(folder_path):
    '''
    folder_path = .../where/your/saved/model/folder
    '''

    meta_paths = sorted(glob.glob(os.path.join(folder_path, '*.meta')))

    numbers = []

    for meta_path in meta_paths:
        numbers.append(int(meta_path.split('-')[-1].split('.')[0]))

    numbers = np.asarray(numbers)

    sorted_idx = np.argsort(numbers)

    latest_meta_path = meta_paths[sorted_idx[-1]]

    ckpt_path = latest_meta_path.replace('.meta', '')
    
    return ckpt_path

def get_ckpt_list(folder_path):
    '''
    folder_path = .../where/your/saved/model/folder
    '''

    meta_paths = sorted(glob.glob(os.path.join(folder_path, '*.meta')))

    numbers = []

    for meta_path in meta_paths:
        numbers.append(int(meta_path.split('-')[-1].split('.')[0]))

    numbers = np.asarray(numbers)

    sorted_idx = np.argsort(numbers)

    ckpt_list = []
    
    for meta_idx in sorted_idx:
        meta_path = meta_paths[meta_idx]
        ckpt_list.append(meta_path.replace('.meta', ''))

    return ckpt_list

def get_image_paths(image_folder):
    possible_image_type = ['jpg', 'png', 'JPEG', 'jpeg']
    
    image_list = [image_path for image_paths in [glob.glob(os.path.join(image_folder, '*.%s' % ext)) for ext in possible_image_type] for image_path in image_paths]

    return image_list

class painter:
    def __init__(self, font_path='./dataset/Arial-Unicode-Bold.ttf'):
        self.font_path = font_path
        self.font = ImageFont.truetype(self.font_path, 20)
        self.label_dict = ['others', 'exterior']

    def write_results(self, images, gt_list, pred_list, how_many=5):
        '''
        images: (batch_size, 224, 224, 3)
        gt_list: (batch_size, 1)
        pred_list: (batch_size, 1), not gone through tf.nn.sigmoid
        '''
        
        batch_size, height, width, c_ch = images.shape

        images = images.astype('uint8')
        
        gt_list = gt_list.astype('int32')
        pred_list = (pred_list >= 0.5).astype('int32')
        
        for idx in range(5):
            img_pil = Image.fromarray(images[idx])
            d_pil = ImageDraw.Draw(img_pil)

            # write gt
            d_pil.text((10, 3), str(self.label_dict[gt_list[idx][0]]), font=self.font, fill=(0,0,255))
            
            # write pred
            d_pil.text((10, 3+20), str(self.label_dict[pred_list[idx][0]]), font=self.font, fill=(255, 0, 0))
            images[idx] = np.asarray(img_pil)

        return images.astype('float32')

##########################################
## rank results along to cosine similarity

def cond(do_idx, input_how_many, input_query, input_features, return_outputs):
    return tf.less(do_idx, input_how_many)

def body(do_idx, input_how_many, input_query, input_features, return_outputs):
    tmp_output = tf.reduce_sum(tf.multiply(input_query[do_idx:do_idx+1], input_features), axis=1)
    return_outputs = return_outputs.write(do_idx, tmp_output)
    do_idx = do_idx + 1

    return do_idx, input_how_many, input_query, input_features, return_outputs

def get_coss_list(input_how_many, input_query, input_features):
    
    
    return_outputs = tf.TensorArray(dtype=tf.float32, size=input_how_many, infer_shape=False)
    _, _, _, _, return_outputs = tf.while_loop(cond, body, [0, input_how_many, input_query, input_features, return_outputs])

    return_outputs = return_outputs.stack()

    return return_outputs
