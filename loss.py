import numpy as np
import tensorflow as tf
import scipy.ndimage as nd
import matplotlib.pyplot as plt
"""
from Loader import Loader
from model import unet
from time import time
"""
CELL = 1


def loss_fn(seg, predictions, weight_map=None):
    """
    loss function according to the Assignment PDF
    with the use of weights as depicted in the Unet paper
    """
    weight_map = weight_map if weight_map is not None else create_weights(seg=seg)
    seg = expand_segmentations(seg)
    loss_map = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=seg)
    weighted_loss = tf.multiply(loss_map, weight_map)
    #weighted_loss = tf.multiply(loss_map, 1)
    weighted_loss = tf.reduce_sum(weighted_loss) / np.shape(seg)[0]
    return weighted_loss


def create_weights(seg, w0=5, sigma=16, visual=False):
    """
    create weights for the loss function
    where w_c is the frequency of each class in the seg
    and bwgt are the weights that mark the border between the adjacent cells
    """
    size = np.shape(seg)
    batch_size = size[0]
    weight_map = np.zeros_like(seg)
    for i in range(batch_size):
        curr_seg = seg[i, :, :, 0]
        uvals = np.unique(curr_seg)  # number of unique variables
        wmp = np.zeros(len(uvals))  # number of classes in Given segmentation
        class_per_pixel = np.zeros((3, size[1], size[2]))

        for uv in range(len(uvals)):
            class_per_pixel[uv] = curr_seg[:, :] == uvals[uv]
            wmp[uv] = 1 / (class_per_pixel[uv]).sum()  # frequency of class
        wmp[:] = wmp[:] / max(wmp[:])  # normalize
        w_c = np.zeros_like(curr_seg)

        for uv in range(len(uvals)):
            w_c[class_per_pixel[uv].astype(int) == 1] = wmp[uv]
        bwgt = 0
        """
        if epoch >= 0:
            s = nd.morphology.generate_binary_structure(2, 2)
            # extract the cells from seg and cluster each instance
            cells, num_features = nd.measurements.label(input=nd.binary_opening(class_per_pixel[CELL]).astype(int),
                                                        structure=s)
            bwgt = np.zeros_like(curr_seg)
            maps = np.zeros((size[1], size[2], num_features))
            if num_features >= 2:
                for ci in range(1, num_features + 1):  # for each instance cell
                    temp = np.array(cells == ci, dtype=np.float32)
                    maps[:, :, ci-1] = nd.morphology.distance_transform_edt(1-temp)
                maps = np.sort(maps, -1)
                d1 = maps[:, :, 0]
                d2 = maps[:, :, 1]
                bwgt = w0 * np.exp(-(np.square(d1 + d2)) / (2 * sigma))
    `   """
        #bwgt = w0 * class_per_pixel[2]
        weight_map[i, :, :, 0] = w_c + bwgt
        if visual:
            plt.subplot(1, 2, 1)
            plt.imshow(curr_seg)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(weight_map.squeeze())
            plt.colorbar()
            plt.show()

    return weight_map.squeeze()


def expand_segmentations(seg):
    size = np.shape(seg)
    expand_seg = np.zeros((size[0], size[1], size[2], 3))
    expand_seg[:, :, :, 0] = seg[:, :, :, 0] == 0
    expand_seg[:, :, :, 1] = seg[:, :, :, 0] == 1
    expand_seg[:, :, :, 2] = seg[:, :, :, 0] == 2
    return expand_seg

