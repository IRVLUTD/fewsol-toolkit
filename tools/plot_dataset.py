#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt


def normalize_descriptor(res, stats=None):
    """
    Normalizes the descriptor into RGB color space
    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor
    """

    if stats is None:
        res_min = res.min()
        res_max = res.max()
    else:
        res_min = np.array(stats['min'])
        res_max = np.array(stats['max'])

    normed_res = np.clip(res, res_min, res_max)
    eps = 1e-10
    scale = (res_max - res_min) + eps
    normed_res = (normed_res - res_min) / scale
    return normed_res


if __name__ == "__main__":

    imgdir = '/capri/Fewshot_Dataset/real_objects'
    save_dir = '../data/images'
    visualize = False

    # list subdirs
    subdirs = sorted(os.listdir(imgdir))
    num = len(subdirs)
    print(subdirs)

    count = 0
    for i in range(0, num, 6):

        dirname1 = subdirs[i]
        dirname2 = subdirs[i+1]
        dirname3 = subdirs[i+2]
        dirname4 = subdirs[i+3]
        dirname5 = subdirs[i+4]
        dirname6 = subdirs[i+5]

        for j in range(9):
            filename1 = os.path.join(imgdir, dirname1, '%06d-color.jpg' % j)
            im1 = cv2.imread(filename1)
            filename2 = os.path.join(imgdir, dirname2, '%06d-color.jpg' % j)
            im2 = cv2.imread(filename2)
            filename3 = os.path.join(imgdir, dirname3, '%06d-color.jpg' % j)
            im3 = cv2.imread(filename3)
            filename4 = os.path.join(imgdir, dirname4, '%06d-color.jpg' % j)
            im4 = cv2.imread(filename4)
            filename5 = os.path.join(imgdir, dirname5, '%06d-color.jpg' % j)
            im5 = cv2.imread(filename5)
            filename6 = os.path.join(imgdir, dirname6, '%06d-color.jpg' % j)
            im6 = cv2.imread(filename6)

            filename1 = os.path.join(imgdir, dirname1, '%06d-label-binary.png' % j)
            mask1 = cv2.imread(filename1)
            filename2 = os.path.join(imgdir, dirname2, '%06d-label-binary.png' % j)
            mask2 = cv2.imread(filename2)
            filename3 = os.path.join(imgdir, dirname3, '%06d-label-binary.png' % j)
            mask3 = cv2.imread(filename3)
            filename4 = os.path.join(imgdir, dirname4, '%06d-label-binary.png' % j)
            mask4 = cv2.imread(filename4)
            filename5 = os.path.join(imgdir, dirname5, '%06d-label-binary.png' % j)
            mask5 = cv2.imread(filename5)
            filename6 = os.path.join(imgdir, dirname6, '%06d-label-binary.png' % j)
            mask6 = cv2.imread(filename6)

            I1 = np.concatenate((im1, mask1), axis=1)
            I2 = np.concatenate((im2, mask2), axis=1)
            I3 = np.concatenate((im3, mask3), axis=1)
            I4 = np.concatenate((im4, mask4), axis=1)
            I5 = np.concatenate((im5, mask5), axis=1)
            I6 = np.concatenate((im6, mask6), axis=1)

            I12 = np.concatenate((I1, I2), axis=1)
            I34 = np.concatenate((I3, I4), axis=1)
            I56 = np.concatenate((I5, I6), axis=1)
            I = np.concatenate((I12, I34, I56), axis=0)
            print(I.shape)

            if visualize:
                # show image
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                plt.imshow(I[:, :, (2, 1, 0)])
                plt.axis('off')
                plt.show()
            else:
                filename = os.path.join(save_dir, '%06d.jpg' % count)
                count += 1
                print(filename)
                cv2.imwrite(filename, I)
