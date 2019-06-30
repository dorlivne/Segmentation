import numpy as np
import tensorflow as tf
import scipy.ndimage as nd
from Loader import *
import matplotlib.pyplot as plt
from model import unet
from time import time

def loss(predictions, seg):
    weight_map = create_weights(seg, visual=False)  # per seg
    seg = expand_segmentations(seg)
    loss_map = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=seg)
    weighted_loss = tf.multiply(loss_map, weight_map)
    return tf.reduce_mean(weighted_loss)


def create_weights(seg, w0=10, sigma=25, visual=False):
    t = time()
    size = np.shape(seg)
    batch_size = size[0]
    weight_map = np.zeros_like(seg)

    for i in range(batch_size):
        curr_seg = seg[i, :, :, 0]
        uvals = np.unique(curr_seg)  # number of unique variables
        wmp = np.zeros(len(uvals))  # number of classes in Given segmentation
        class_per_pixel = np.zeros((3, size[1], size[2]))
        for uv in range(len(uvals)):
            class_per_pixel[uv] = (curr_seg[:, :] == uvals[uv])
            wmp[uv] = 1 /class_per_pixel[uv].sum()  # frequency of class
        wmp[:] = wmp[:] / max(wmp[:])  # normalize
        w_c = np.zeros_like(curr_seg)

        for uv in range(len(uvals)):
            w_c[curr_seg[:, :] == uvals[uv]] = wmp[uv]

        # extract the cells from seg and cluster each instance
        cells, num_features = nd.measurements.label(input=(curr_seg[:, :] == 1))

        bwgt = np.zeros_like(curr_seg)
        maps = np.zeros((size[1], size[2], num_features))
        if num_features >= 2:
            for ci in range(num_features):  # for each instance cell
                temp = np.array(cells == ci,dtype=bool)
                temp = temp.astype(dtype=int)
                maps[:, :, ci] = nd.morphology.distance_transform_edt(1-temp)  # distance from cell ci for each pixel

            maps = np.sort(maps, -1)
            d1 = maps[:, :, 0]
            d2 = maps[:, :, 1]
            bwgt = w0 * np.exp(-(np.square(d1 + d2))/(2 * sigma))
            bwgt = np.multiply(bwgt, cells == 0)
        weight_map[i, :, :, 0] = w_c + bwgt

        if visual:
            plt.subplot(1, 2, 1)
            plt.imshow(curr_seg)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(weight_map[i, :, :, 0])
            plt.colorbar()
            plt.show()
    elapsed_time = time() - t
    print("elapsed_time is {}".format(elapsed_time))
    return weight_map


def expand_segmentations(seg):
    SIZE = np.shape(seg)
    expand_seg = np.zeros((SIZE[0], SIZE[1], SIZE[2], 3))
    expand_seg[:, :, :, 0] = seg[:, :, :, 0] == 0
    expand_seg[:, :, :, 1] = seg[:, :, :, 0] == 1
    expand_seg[:, :, :, 2] = seg[:, :, :, 0] == 2
    return expand_seg


if __name__ == '__main__':
     loader = Loader(batch_size=1)
     image, seg = loader.get_one_batch()
     unet_model = unet()
     predictions = unet_model(image)
     a = loss(predictions=predictions, seg=seg)