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
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--car_folder')
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
    images = sorted(glob.glob(os.path.join(args.car_folder, '*.jpg')))
    
    label_path = os.path.join(args.car_folder, 'cars_annos.mat')
    
    cars_annos = loadmat(label_path)
    
    #print(cars_annos)   

    annotations = cars_annos['annotations'].ravel()
    
    annotations = sorted(annotations, key=lambda a: str(a[0][0]))
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    split_basis = 99 # before 99, train, after 99, test

    crop = True
    for annotation in annotations:
        class_label = int(annotation[5][0][0])
        file_name = annotation[0][0].replace('car_ims/', '')
        from_path = os.path.join(args.car_folder, file_name)

        if class_label < split_basis:
            mode = 'train'
        else:
            mode = 'test'

        tmp_sub_folder = os.path.join(args.save_path, mode)
        if not os.path.exists(tmp_sub_folder):
            os.mkdir(tmp_sub_folder)

        sub_folder = os.path.join(args.save_path, mode, '%05d' % (class_label))
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
        
        to_path = os.path.join(sub_folder, file_name)
        if crop:
            img = cv2.imread(from_path)

            x = int(annotation[1][0][0])
            y = int(annotation[2][0][0])
            w = int(annotation[3][0][0])
            h = int(annotation[4][0][0])

            img = img[y:h, x:w]
            cv2.imwrite(to_path, img)
        else:
            shutil.copy(from_path, to_path) 
        
        #print(from_path)
        #print(to_path)
