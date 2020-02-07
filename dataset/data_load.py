'''
symmetrical-synthesis
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
import os
import time
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
import random
try:
    import data_util
except ImportError:
    from dataset import data_util

tf.app.flags.DEFINE_boolean('random_resize', False, 'True or False')
tf.app.flags.DEFINE_boolean('past_dataset', False, 'True or False')
tf.app.flags.DEFINE_string('google_path', None, '')
tf.app.flags.DEFINE_integer('min_train3', 2, '')
tf.app.flags.DEFINE_string('match_info', None, '')
tf.app.flags.DEFINE_float('match_prob', 0.0, '')
tf.app.flags.DEFINE_boolean('mnist_mode', False, '')
FLAGS = tf.app.flags.FLAGS

'''
image_path = '/where/your/images/*.jpg'
'''

def load_image(im_fn, input_size=224):
    org_image = cv2.imread(im_fn, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:,:,::-1] # rgb converted
    '''
    if FLAGS.random_resize:
        resize_table = [0.5, 1.0, 1.5, 2.0]
        selected_scale = np.random.choice(resize_table, 1)[0]
        shrinked_hr_size = int(hr_size / selected_scale)

        h, w, _ = high_image.shape
        if h <= shrinked_hr_size or w <= shrinked_hr_size:
            high_image = cv2.resize(high_image, (hr_size, hr_size))
        else:
            h_edge = h - shrinked_hr_size
            w_edge = w - shrinked_hr_size
            h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
            w_start = np.random.randint(low=0, high=w_edge, size=1)[0]
            high_image_crop = high_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]
            high_image = cv2.resize(high_image_crop, (hr_size, hr_size))
    '''
    
    h, w, _ = org_image.shape
    
    min_len = np.min([h, w])

    # center crop margin, we follow the method, which was introduced in DELF paper.
    
    if FLAGS.mnist_mode:
        crop_image = org_image.copy()
    else:
        try:
            cc_margin = np.random.randint(low=1, high=int(min_len * 0.05), size=1)[0]
            crop_image = org_image[cc_margin:-cc_margin, cc_margin:-cc_margin, :].copy()
        except:
            crop_image = org_image.copy()
    
    new_input_size = int(input_size * 1.125)
    crop_image = cv2.resize(crop_image, (new_input_size, new_input_size), interpolation=cv2.INTER_AREA)
    # random crop range

    h_edge = new_input_size - input_size#32#256 - input_size # input_size is 224
    w_edge = new_input_size - input_size#256 - input_size
    
    h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
    w_start = np.random.randint(low=0, high=w_edge, size=1)[0]

    return_image = crop_image[h_start:h_start+input_size, w_start:w_start+input_size,:]

    # flip lr
    if random.randint(0, 1):
        return_image = return_image[:,::-1,:]
    #print('return', return_image.shape)
    return return_image #high_image, low_image

def get_images_dict(image_folder):
    '''
    image_folder = '/data/IR/DB/sid_images'
    folder structure
    sid_images     - sid0 - image00.png, image01.png, ...
                   - sid1 - ...
                   - sid2 - ...
    '''
    if FLAGS.match_info is not None:
        match_dict = {}

        f_match = open(FLAGS.match_info, 'r')
        match_lines = f_match.readlines()

        cnt = 0
        for match_line in match_lines:
            ver1_cls, ver2_cls, prob = match_line.split()
            prob = float(prob)
            if prob >= FLAGS.match_prob:
                match_dict[ver2_cls] = 1
    possible_image_type = ['jpg', 'JPG', 'png', 'JPEG', 'jpeg']

    sid_list = glob.glob(os.path.join(image_folder, '*'))
    
    images_dict = {}
    images_list = []   
    images_cnt = 0

    sid_idx = 0
    for sid_folder in sid_list:
        ext_folder = sid_folder
        #ext_folder = os.path.join(sid_folder, 'exterior')
        images_path = [image_path for image_paths in [glob.glob(os.path.join(ext_folder, '*.%s' % ext)) for ext in possible_image_type] for image_path in image_paths]    
        
        n_instance = 2
        
        if len(images_path) < n_instance:
            continue
        
        for image_path in images_path:
            images_list.append([image_path, sid_idx])

        images_dict[sid_idx] = images_path
        images_cnt += len(images_path)
        sid_idx += 1
    #print(images_dict)
    
    stat_db = {}
    stat_db['num_sid'] = len(images_dict)
    stat_db['images_cnt'] = images_cnt

    return images_dict, stat_db, images_list

def get_record(image_folder, input_size, batch_size):
    images_dict, stat_db, images_list = get_images_dict(image_folder)
    print('place total sids: %d, total images: %d' % (stat_db['num_sid'], stat_db['images_cnt']))
    if FLAGS.google_path is not None:
        images_dict_google, stat_db_google, images_list_google = get_images_dict(FLAGS.google_path)
        print('google total sids: %d, total images: %d' % (stat_db_google['num_sid'], stat_db_google['images_cnt']))
    #time.sleep(3)
    
    n_instance = 2
    b_replace = False
    real_batch_size = batch_size // n_instance
    while True:
        try:
            gt_labels = np.random.choice(len(images_dict), real_batch_size, replace=b_replace)
    
            anchor_images = []
            pos_images = []
            
            for n in range(n_instance - 1):
                pos_images.append([])
            
            for label in gt_labels:
                tmp_image_list = images_dict[label]
                image_index = np.random.choice(len(tmp_image_list), n_instance, replace=False)
                
                anchor_image = load_image(tmp_image_list[image_index[0]], input_size)
                
                anchor_images.append(anchor_image)

                for n, ind in enumerate(image_index[1:]):
                    pos_image = load_image(tmp_image_list[ind], input_size)
                    pos_images[n].append(pos_image)
    
            #print(len(gt_labels))
            if n_instance == 2:
                pos_images = pos_images[0]
            elif n_instance == 1:
                pos_images = pos_images
            else:
                pos_images = np.concatenate(pos_images, axis=0)

            yield anchor_images, pos_images, gt_labels #im_fn, gt_label
        except Exception as e:
            print(e)
            continue

def generator(image_folder, input_size=224, batch_size=32):
    for anchor_images, pos_images, gt_labels in get_record(image_folder, input_size, batch_size):
        yield anchor_images, pos_images, gt_labels


def get_generator(image_folder, **kwargs):
    return generator(image_folder, **kwargs)

## image_path = '/where/is/your/images/'
def get_batch(image_path, num_workers, **kwargs):
    try:
        generator = get_generator(image_path, **kwargs)
        enqueuer = data_util.GeneratorEnqueuer(generator, use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_ouptut = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.001)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == '__main__':
    image_path = '/data/IR/DB/data_refinement/place_exterior'
    num_workers = 4
    batch_size = 128
    input_size = 224

    data_generator = get_batch(image_path=image_path,
                               num_workers=num_workers,
                               batch_size=batch_size,
                               input_size=224)
    
    _ = 0
    while True:
        _ += 1
        #break
        start_time = time.time()
        data = next(data_generator)
        anchor_images = np.asarray(data[0])
        pos_images = np.asarray(data[1])
        gts = np.asarray(data[2])
        print('%d done!!! %f' % (_, time.time() - start_time), anchor_images.shape, pos_images.shape, gts.shape)
        #for sub_idx, (loaded_image, gt) in enumerate(zip(loaded_images, gts)):
        #    save_path = '/data/IR/DB/naver_place/test/%03d_%03d_gt_%d_image.jpg' % (_, sub_idx, gt)
        #    cv2.imwrite(save_path, loaded_image[:,:,::-1])
