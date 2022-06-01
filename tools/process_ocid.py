#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy
import shutil
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path
from PIL import Image
from mask import imread_indexed, visualize_segmentation


def process_label(foreground_labels):
    unique_nonnegative_indices = np.unique(foreground_labels)
    mapped_labels = foreground_labels.copy()
    for k in range(unique_nonnegative_indices.shape[0]):
        mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
    foreground_labels = mapped_labels
    return foreground_labels


def list_dataset(ocid_object_path):
    data_path = Path(ocid_object_path)
    seqs = sorted(list(Path(data_path).glob('**/*seq*')))
    return seqs


def save_data(save_dir, object_id, im, im_label, im_box, meta):

    seq_name = meta['seq_name']
    image_name = meta['image_name'][:-4]
    folder_name = seq_name.replace('/', '_') + '_obj%02d' % object_id
    print(folder_name, image_name)

    filename = os.path.join(save_dir, folder_name)
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    filename = os.path.join(save_dir, folder_name, image_name + '-color.jpg')
    cv2.imwrite(filename, im)

    filename = os.path.join(save_dir, folder_name, image_name + '-label.jpg')
    cv2.imwrite(filename, im_label)

    filename = os.path.join(save_dir, folder_name, image_name + '-box.jpg')
    cv2.imwrite(filename, im_box)

    filename = os.path.join(save_dir, folder_name, image_name + '-meta.mat')
    scipy.io.savemat(filename, meta, do_compression=True)


if __name__ == "__main__":
    dataset_dir = '../data/OCID/'
    save_dir = '../data/OCID_objects/'
    is_show = True

    # list subdirs
    seqs = list_dataset(dataset_dir)
    num_seqs = len(seqs)
    print('%d seqs' % num_seqs)

    # for each sequence
    for i in range(num_seqs):
        seq_name = seqs[i]
        print(i, seq_name)

        # list images
        rgb_folder = os.path.join(seq_name, 'rgb')
        image_names = sorted(os.listdir(rgb_folder))
        print(image_names)

        object_id = 1
        for j in range(len(image_names)):
            # read image
            filename = os.path.join(seq_name, 'rgb', image_names[j])
            im = cv2.imread(filename)

            # read mask
            label_filename = filename.replace('rgb', 'label')
            label = imread_indexed(label_filename)

            # mask table as background
            label[label == 1] = 0
            if 'table' in label_filename:
                label[label == 2] = 0
            label = process_label(label)

            if len(np.unique(label)) == 1:
                continue

            # label image
            mask = np.zeros_like(label)
            max_label = np.max(label)
            mask[label == max_label] = 1
            im_label = visualize_segmentation(im, mask, return_rgb=True)

            # bounding box image
            im_box = im.copy()
            y, x = np.nonzero(mask)
            if len(x) > 0 and len(y) > 0:
                x1 = np.min(x)
                x2 = np.max(x)
                y1 = np.min(y)
                y2 = np.max(y)
                im_box = cv2.rectangle(im_box, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)

            # meta data
            meta = {'seq_name': str(seq_name).replace(dataset_dir, ''), 'image_name': image_names[j]}
            meta['object_id'] = object_id
            meta['max_label'] = max_label
            meta['label'] = label

            if is_show:
                fig = plt.figure()
                ax = fig.add_subplot(2, 2, 1)
                plt.imshow(im[:, :, (2, 1, 0)])
                ax.set_title('image')
                plt.axis('off')

                ax = fig.add_subplot(2, 2, 2)
                plt.imshow(im_label)
                ax.set_title('image labeled')
                plt.axis('off')

                ax = fig.add_subplot(2, 2, 3)
                plt.imshow(im_box[:, :, (2, 1, 0)])
                ax.set_title('image box')
                plt.axis('off')

                ax = fig.add_subplot(2, 2, 4)
                plt.imshow(label)
                ax.set_title('label')
                plt.axis('off')
                plt.show()
            else:
                # save data
                save_data(save_dir, object_id, im, im_label, im_box, meta)
            object_id += 1
