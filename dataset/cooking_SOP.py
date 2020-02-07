'''
symmetrical-synthesis
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
import os
import sys
import glob
import cv2
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--sop_folder')
parser.add_argument('--save_path')
args = parser.parse_args()

def copy_images(info_lines, mode='train'):
    train_folder = os.path.join(args.save_path, mode)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    for info_line in info_lines[1:]:
        image_id, class_id, super_class, file_path = info_line.split()
        
        sub_folder = os.path.join(train_folder, '%s' % (class_id))
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)

        from_image = os.path.join(args.sop_folder, file_path)
        to_image = os.path.join(sub_folder, os.path.basename(file_path))

        shutil.copy(from_image, to_image)


if __name__ == '__main__':
    # IR process!!!
    
    train_info_path = os.path.join(args.sop_folder, 'Ebay_train.txt')
    f_train_info = open(train_info_path, 'r')
    train_info_lines = f_train_info.readlines()
    
    test_info_path = os.path.join(args.sop_folder, 'Ebay_test.txt')
    f_test_info = open(test_info_path, 'r')
    test_info_lines = f_test_info.readlines()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    copy_images(train_info_lines, 'train')
    copy_images(test_info_lines, 'test')



