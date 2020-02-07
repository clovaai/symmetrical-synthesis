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
import glob
import cv2
import collections
import shutil
import h5py
import math
import random
import tqdm

tf.app.flags.DEFINE_string('run_gpu', '0', 'use single gpu')
tf.app.flags.DEFINE_string('model_path', '/where/your/trained/model/folder', '')
tf.app.flags.DEFINE_string('image_path', '/where/your/saved/image/folder', '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_integer('dim_features', 256, '')
tf.app.flags.DEFINE_string('log_path', '/save/log/folder', '')
tf.app.flags.DEFINE_integer('start_idx', 0, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, 'imagenet/pretrained_model.ckpt')
tf.app.flags.DEFINE_boolean('center_crop', False, 'True or False')
tf.app.flags.DEFINE_float('sleep_time', 1.0, '')
tf.app.flags.DEFINE_string('k_list', '1,2,3,4,5', '')
tf.app.flags.DEFINE_boolean('eval_once', False, '')

FLAGS = tf.app.flags.FLAGS

def load_models(model_path, len_image_list, max_k):
    input_images = tf.placeholder(tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3], name='input_images')

    model_builder = models.model_builder(batch_size=FLAGS.batch_size, n_classes=FLAGS.num_classes, dim_features=FLAGS.dim_features, is_training=False)

    # Extract L2 normalized features
    output_features, cnn_features = model_builder.build_features_extractor_test(input_images)
    
    #### cosine similarity
    input_query_features = tf.placeholder(tf.float32, shape=[None, FLAGS.dim_features], name='input_query')
    input_extracted_features = tf.placeholder(tf.float32, shape=[None, FLAGS.dim_features], name='input_features')
    
    coss_list = tf.tensordot(input_query_features, tf.transpose(input_extracted_features), axes=1)
    
    top_k_values, top_k_indices = tf.nn.top_k(coss_list, k=max_k + 1, sorted=True)
    
    # model loader
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    saver = tf.train.Saver(all_vars)
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    #ckpt_list = utils.get_ckpt_list(model_path)

    #saver.restore(sess, ckpt_path)
    
    return input_images, output_features, sess, input_query_features, input_extracted_features, coss_list, top_k_indices, top_k_values, saver, cnn_features
    #ckpt_list = utils.get_ckpt_list(model_path)
    #return input_images, pred, sess, saver, ckpt_list
   

def load_image_paths(image_path, sid_image_dict=None, image_sid_dict=None, image_batch=None):
    '''
    image_path - sid0 - image0, image1, ..., q_image0, q_image1
               - sid1 - ...
               - ...
    '''
    # get positive images

    sid_folders = glob.glob(os.path.join(image_path, '*'))

    if sid_image_dict is None:
        sid_image_dict = collections.OrderedDict() # key: sid, values: images path
    
        image_sid_dict = collections.OrderedDict() # key: image path, value: sid
    
        image_batch = []

    for sid_idx in tqdm.tqdm(range(len(sid_folders))):
        sid_folder = sid_folders[sid_idx]
#        if sid_idx % 5 == 0:
#            print('loading images, %d / %d' % (sid_idx + 1, len(sid_folders)))
        sid = os.path.basename(sid_folder)
        
        sid_images = glob.glob(os.path.join(sid_folder, '*'))
        
        if len(sid_images) == 1:
            continue

        for image_path in sid_images:
                
            # load image and check fail
            try:
                loaded_image = cv2.imread(image_path)
                
                if FLAGS.center_crop:
                    loaded_image = cv2.resize(loaded_image, (256, 256), interpolation=cv2.INTER_AREA)
                    height, width, _ = loaded_image.shape
                    offset_y = (height - FLAGS.input_size) // 2
                    offset_x = (width - FLAGS.input_size) // 2

                    if (height - FLAGS.input_size) % 2 and (width - FLAGS.input_size) % 2:
                        loaded_image = loaded_image[offset_y+1:-offset_y, offset_x+1:-offset_x,:]
                        #print('if', loaded_image.shape, height, width)
                    else:
                        loaded_image = loaded_image[offset_y:-offset_y, offset_x:-offset_x,:]
                        #print('else', loaded_image.shape, height, width)

                else:
                    loaded_image = cv2.resize(loaded_image, (FLAGS.input_size, FLAGS.input_size), interpolation=cv2.INTER_AREA)
                image_batch.append(loaded_image[:,:,::-1])

                image_name = os.path.basename(image_path)
            
                image_sid_dict[image_path] = sid
                
                if sid not in sid_image_dict:
                    sid_image_dict[sid] = []
                    sid_image_dict[sid].append(image_path.split('/')[-2])
                else:
                    sid_image_dict[sid].append(image_path.split('/')[-2])
            except Exception as e:
                print('image load fail', image_path)
    
    print('\n%d sids and %d images\n' % (len(sid_image_dict), len(image_sid_dict)))

    return sid_image_dict, image_sid_dict, image_batch

def load_images(image_list):
    image_batch = []
    for image in image_list:
        try:
            loaded_image = cv2.imread(image)
            loaded_image = cv2.resize(loaded_image, (FLAGS.input_size, FLAGS.input_size), interpolation=cv2.INTER_AREA)

            image_batch.append(loaded_image[:,:,::-1])
        except Exception as e:
            print('image load fail', image)

    return np.asarray(image_batch)

def extract_features(input_images, output_features, sess, loaded_images):
    how_many = len(loaded_images) // FLAGS.batch_size
    features = np.zeros([len(loaded_images), FLAGS.dim_features], dtype='float32')
    
    for idx in range(how_many + 1):
        last_bttn = False
        if (idx + 1) * FLAGS.batch_size >= len(loaded_images):
            selected_image = loaded_images[idx*FLAGS.batch_size:]
            last_bttn = True
        else:
            selected_image = loaded_images[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
        #print(idx * FLAGS.batch_size, len(selected_image), len(image_list))
        if len(selected_image) == 0:
            break
        image_batch = selected_image #load_images(selected_image)
        extracted_features = sess.run(output_features, feed_dict={input_images: image_batch})
        
        if last_bttn:
            features[idx*FLAGS.batch_size:,:] = extracted_features.copy()
        else:
            features[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size,:] = extracted_features.copy()
    return features

def get_top_k_indices(input_query_features, input_extracted_features, top_k_indices, top_k_values, sess, query_features, index_features):
    num_features = len(query_features)
    
    check_size = 10000

    how_many = num_features // check_size

    sorted_idx = []

    for idx in range(how_many + 1):
        if (idx + 1) * check_size >= num_features:
            selected_query_features = query_features[idx * check_size:]
        else:
            selected_query_features = query_features[idx * check_size:(idx+1) * check_size]
        
        feed_dict = {input_query_features: selected_query_features,
                     input_extracted_features: index_features}
        
        sorted_idx.append(sess.run(top_k_indices, feed_dict=feed_dict))
    
    output_top_k = np.concatenate(sorted_idx, axis=0)

    return output_top_k
 
#########
def get_mAP(sid_image_dict, image_list, output_top_k):
    # mAP, Recall@1, 3, 5

    ## for test
    ap_list = []
    
    for query_idx, top_k in enumerate(output_top_k):
        query_image_path = image_list[query_idx]
        query_sid = query_image_path.split('/')[-2]
        
        gt_pos_list = sid_image_dict[query_sid]
        num_gt_pos_list = np.min([len(gt_pos_list) - 1, 5]) # 5 was FLAGS.top_k

        old_recall = 0.0
        old_precision = 1.0
        ap = 0.0

        intersect_size = 0

        j = 0
        pred_pos_list = []

        for pos_idx in top_k[1:]:
            pos_image_path = image_list[pos_idx]
            pos_sid = pos_image_path.split('/')[-2]
            pred_pos_list.append(pos_sid)
            
            if pos_sid == query_sid:
                intersect_size += 1

            recall = intersect_size / float(num_gt_pos_list)
            precision = intersect_size / (j + 1.0)

            ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
            if intersect_size == num_gt_pos_list:
                break

            old_recall = recall
            old_precision = precision
            j += 1
        
        ap_list.append(ap)
        
        #print(query_sid, pred_pos_list[:5])
    
    mAP = np.mean(np.asarray(ap_list))
    return mAP

def get_recall(sid_image_dict, image_list, output_top_k, k_list=[1, 3, 5]):

    count_dict = {k:0.0 for k in k_list} # we only support recall@1, 3, 5

    for query_idx, top_k in enumerate(output_top_k):
        top_k = list(filter(lambda x: x!= query_idx, top_k))
        
        sorted_labels = [image_list[pos_idx].split('/')[-2] for pos_idx in top_k]
        query_image_path = image_list[query_idx]
        query_label = query_image_path.split('/')[-2]

        for k in k_list:
            if query_label in sorted_labels[:k]:
                count_dict[k] += 1.
    
    num_queries = len(output_top_k)
    
    for k in k_list:
        count_dict[k] /= num_queries
    
    return count_dict

def init_resize_image(im, maximum_size=512):
    h, w, _ = im.shape
    size = [h, w]
    max_arg = np.argmax(size)
    max_len = size[max_arg]
    min_arg = max_arg - 1
    min_len = size[min_arg] 
    if max_len < maximum_size:
        maximum_size = max_len
        ratio = 1.0
        return im
    else:
        ratio = maximum_size / max_len
        max_len = max_len * ratio 
        min_len = min_len * ratio 
        size[max_arg] = int(max_len)
        size[min_arg] = int(min_len)
        im = cv2.resize(im, (size[1], size[0]), interpolation = cv2.INTER_AREA)
        return im

def get_symm_pt(anchor, pos):
    P = pos
    A = anchor
    U = A / np.linalg.norm(A, axis=1, keepdims=True)
    R = A + np.sum(np.multiply((P - A), U), axis=1, keepdims=True) * U
    Q = 2.0 * R - P
    return Q

def get_symm_pts(l2_whole_features, label_list, unique_label_list):
    n_features, n_dims = l2_whole_features.shape
    
    symm_features = np.zeros([n_features, n_dims], dtype='float32')

    start_idx = 0
    label_list = np.array(label_list)
    for unique_label in unique_label_list:
        selected_features = l2_whole_features[label_list == unique_label]
        rolled_features = np.roll(selected_features, -1, axis=0)
        if random.randint(0, 1):
            sym = get_symm_pt(selected_features, rolled_features)
        else:
            sym = get_symm_pt(rolled_features, selected_features)
        symm_features[start_idx:start_idx + len(selected_features),:] = sym
        start_idx += len(selected_features)
    
    return symm_features

def main(argv=None):
    ######################### System setup
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.run_gpu
    
    ######################### load images
    # load real eval images
    sid_image_dict, image_sid_dict, total_image_batch = load_image_paths(FLAGS.image_path)
    
    len_real_eval = len(total_image_batch)
    #FLAGS.top_k = len_real_eval # rank whole features!
    
    # load distractors
    total_image_batch = np.asarray(total_image_batch)

    # pre load_images
    image_list = list(image_sid_dict.keys())
    
    n_class = len(sid_image_dict.keys())
    
    label_list = []#list(sid_image_dict.keys())
    for image_path in image_list:
        label_list.append(int(image_path.split('/')[-2].split('.')[0]))

    unique_idx = np.unique(label_list, return_index=True)[1]
    unique_label_list = [label_list[idx] for idx in sorted(unique_idx)]

    print('real eval: %d images, distractors: %d images loaded!!!' % (len_real_eval, len(total_image_batch) - len_real_eval))
   
    # setup k_list

    k_list = [int(k) for k in FLAGS.k_list.split(',')]

    max_k = np.max(k_list)

    ######################### Model setup
    input_images, output_features, sess, input_query_features, input_extracted_features, coss_list, top_k_indices, top_k_values, saver, cnn_features = load_models(FLAGS.model_path, len(image_list), max_k)
   
    model_folders = FLAGS.model_path.split(',') 
    
    ######################### log setup
    if not os.path.exists(FLAGS.log_path):
        os.mkdir(FLAGS.log_path)
    
    
    input_dict = {}
    for k in k_list:
        input_dict['Recall@%d' % k] = tf.placeholder(tf.float32, name='input_r%d' % k)
        tf.summary.scalar('Recall@%d' % k, input_dict['Recall@%d' % k])

    summary_op = tf.summary.merge_all()

    summary_writer_dict = {}
    ckpt_done_dict = {}
    best_mAP_dict = {}
    for model_folder in model_folders:
        folder_name = os.path.basename(model_folder)
           
        sub_log_path = os.path.join(FLAGS.log_path, folder_name)
        if not os.path.exists(sub_log_path):
            os.mkdir(sub_log_path)
        
        backup_model_folder = os.path.join(model_folder, 'backup')
        if not os.path.exists(backup_model_folder):
            os.mkdir(backup_model_folder)  ## best model will be saved here

        summary_writer_dict[folder_name] = tf.summary.FileWriter(sub_log_path)
        
        ckpt_done_dict[folder_name] = []
        best_mAP_dict[folder_name] = 0.0
    
    while True:
        for model_folder in model_folders:
            folder_name = os.path.basename(model_folder)

            print('now checking,', folder_name)
            ckpt_list = utils.get_ckpt_list(model_folder)
            summary_writer = summary_writer_dict[folder_name]
            
            backup_model_folder = os.path.join(model_folder, 'backup')

            for ckpt_path in ckpt_list[FLAGS.start_idx:]:
                
                if ckpt_path in ckpt_done_dict[folder_name]:
                    print(ckpt_path, 'already done')
                    continue
                try:
                    saver.restore(sess, ckpt_path) 
                    print('%s model loaded!!!' % (ckpt_path))
                    n_iter = int(ckpt_path.split('-')[-1])
            
                    ######################### Extract features
                    # extract whole features
                    whole_features = extract_features(input_images, output_features, sess, total_image_batch)
                    
                   ######################### Get top_k indices
                    l2_whole_features = whole_features
                    output_top_k = get_top_k_indices(input_query_features, input_extracted_features, top_k_indices, top_k_values, sess, l2_whole_features, l2_whole_features)
                    print(output_top_k.shape)               
                    
                    recall_dict = get_recall(sid_image_dict, image_list, output_top_k, k_list)
                    
                    result_info = ''
                    for k in k_list:
                        result_info += 'Recall@%d: %.4f, ' % (k, recall_dict[k])
                        
                    result_info = result_info[:-2] + '\n'
                    
                    print(result_info)
                    
                    ### do h5py
                    if FLAGS.eval_once:
                        print(recall_dict)
                        exit()

                   
                    ## update summary
                    summary_feed_dict = {}
                    for k in k_list:
                        summary_feed_dict[input_dict['Recall@%d' % k]] = recall_dict[k]
                    
                    summary_str = sess.run(summary_op, feed_dict=summary_feed_dict)
                    
                    summary_writer.add_summary(summary_str, global_step=n_iter)
                    summary_writer.flush()

                    ckpt_done_dict[folder_name].append(ckpt_path)

                    ## check best model
                    
                    #print(mAP, best_mAP_dict[folder_name])
                    mAP = recall_dict[k_list[0]]#r1
                    if mAP > best_mAP_dict[folder_name]:
                        best_mAP_dict[folder_name] = mAP
                        saved_in_backup = glob.glob(os.path.join(backup_model_folder, '*'))
                        for saved_file in saved_in_backup:
                            os.remove(saved_file)

                        best_model_files = glob.glob(ckpt_path + '*')
                        #print(best_model_files)
                        for best_model_file in best_model_files:
                            print(best_model_file, 'saved')
                            shutil.copy(best_model_file, backup_model_folder)

                except Exception as e:
                    print(e)
                    pass

        print(best_mAP_dict)
        print('sleep')
        time.sleep(FLAGS.sleep_time)

if __name__ == '__main__':
    tf.app.run()
