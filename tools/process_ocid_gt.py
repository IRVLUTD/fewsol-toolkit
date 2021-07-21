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


def save_data(save_dir, im_label, image_name):

    filename = os.path.join(save_dir, image_name[:-4] + '-label.jpg')
    cv2.imwrite(filename, im_label)


if __name__ == "__main__":
    dataset_dir = '../data/OCID/'
    save_dir = '../data/OCID_segmentations/'
    is_show = False

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

            im_label = visualize_segmentation(im, label, return_rgb=True)

            # save data
            save_data(save_dir, im_label, image_names[j])

            if is_show:
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(im[:, :, (2, 1, 0)])
                ax.set_title('image')
                plt.axis('off')

                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(im_label)
                ax.set_title('image labeled')
                plt.axis('off')

                plt.show()
