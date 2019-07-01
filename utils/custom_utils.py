from loss import create_weights, jaccard_score, tf
import numpy as np

def extract_weight_map(queue_seg, queue_weightmap):
    seg = queue_seg.get()
    queue_weightmap.put(create_weights(seg=seg))
    extract_weight_map(queue_seg, queue_weightmap)


# TODO implement this function
def Jaccard_Index(output_batch, gt_batch, channel_axis=-1, foreground_class_index=1):
    gt_batch = tf.squeeze(gt_batch, channel_axis)
    gt_foreground = (tf.equal(tf.cast(gt_batch, tf.float32), foreground_class_index).numpy()).astype(np.int8)
    output_classes = tf.argmax(output_batch, axis=channel_axis)
    output_foreground = (tf.equal(output_classes, foreground_class_index).numpy()).astype(np.int8)
    return jaccard_score(gt_foreground.flatten(), output_foreground.flatten())

