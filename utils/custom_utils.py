from loss import create_weights, tf, np
#from sklearn.metrics import jaccard_score


def extract_weight_map(queue_seg, queue_weightmap):
    while True:
        seg = queue_seg.get()
        try:
            if not seg:
                return
        except Exception:
            pass
        queue_weightmap.put(create_weights(seg=seg))

"""
def Jaccard_Index2(output_batch, gt_batch, channel_axis=-1, foreground_class_index=1):
    gt_batch = tf.squeeze(gt_batch, channel_axis)
    gt_foreground = (tf.equal(tf.cast(gt_batch, tf.float32), foreground_class_index).numpy()).astype(np.int8)
    output_classes = tf.argmax(output_batch, axis=channel_axis)
    output_foreground = (tf.equal(output_classes, foreground_class_index).numpy()).astype(np.int8)
    return jaccard_score(gt_foreground.flatten(), output_foreground.flatten()
"""


def Jaccard_Index(gt_batch, output_batch, channel_axis=-1, foreground_class_index=1):
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
    return seg_measure