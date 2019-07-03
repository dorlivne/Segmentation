from loss import create_weights, tf
import numpy as np
import matplotlib.pyplot as plt

def extract_weight_map(queue_seg, queue_weightmap):
    seg = queue_seg.get()
    queue_weightmap.put(create_weights(seg=seg))
    extract_weight_map(queue_seg, queue_weightmap)


# TODO implement this function
def Jaccard_Index(gt_batch, output_batch, channel_axis=-1, foreground_class_index=1,visual=False):
    spatial_dimensions = (1, 2)
    gt_batch = tf.squeeze(gt_batch, channel_axis)
    gt_foreground = (tf.equal(tf.cast(gt_batch, tf.float32), foreground_class_index).numpy()).astype(np.int8)
    output_classes = tf.argmax(output_batch, axis=channel_axis)
    output_foreground = (tf.equal(output_classes, foreground_class_index).numpy()).astype(np.int8)
    intersection = tf.cast(tf.math.count_nonzero(tf.logical_and(gt_foreground, output_foreground),
                                                 axis=spatial_dimensions), tf.float32)
    union = tf.cast(tf.math.count_nonzero(tf.logical_or(gt_foreground, output_foreground),
                                          axis=spatial_dimensions), tf.float32)
    area = tf.cast(tf.math.count_nonzero(gt_foreground, axis=spatial_dimensions), tf.float32)
    num_objects = tf.cast(tf.math.count_nonzero(tf.greater(area, 0)), tf.float32)
    jaccard = tf.cast(tf.divide(intersection, union + 0.00000001), tf.float32)
    seg_measure = tf.reduce_sum(jaccard)/num_objects
    if visual:
        plt.subplot(1, 2, 1)
        plt.imshow(gt_foreground[0])
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(output_foreground[0])
        plt.colorbar()
        plt.show()

    return seg_measure
