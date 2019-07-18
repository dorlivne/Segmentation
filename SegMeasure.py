import tensorflow as tf
from scipy import ndimage
import numpy as np

__author__ = 'arbellea@post.bgu.ac.il'


def SegMeasure(channel_axis=-1, foreground_class_index=1):
    spatial_dimensions = (1, 2) if channel_axis == -1 else (2, 3)

    strel = np.zeros([3, 3])
    strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def connencted_components(input_np):
        labeled = np.zeros_like(input_np, dtype=np.uint16)
        max_num = 0
        for d, image in enumerate(input_np):
            labeled_image, max_num_temp = ndimage.label(image, structure=strel)
            labeled[d] = labeled_image
            max_num = np.maximum(max_num, max_num_temp)

        return labeled, np.array(max_num).astype(np.float32)

    def seg_numpy(gt, seg):
        gt_labled, _ = connencted_components(gt.numpy())
        seg_labled, _ = connencted_components(seg.numpy())

        all_iou = []
        for gt, seg in zip(gt_labled, seg_labled):

            for l in np.unique(gt):
                all_iou.append(0.)
                bw = gt == l
                l_area = np.sum(bw)
                if l == 0:
                    continue
                overlaping_inds = seg[bw]
                for s in np.unique(overlaping_inds):
                    if s == 0:
                        continue
                    intersection = np.sum(overlaping_inds == s)
                    overlap = intersection / l_area
                    if overlap > 0.5:
                        s_area = np.sum(seg == s)
                        iou = intersection / (l_area + s_area - intersection)
                        all_iou[-1] = iou
        return np.mean(all_iou)

    # @tf.function
    def calc_seg(gt, net_output):
        with tf.device('/cpu:0'):
            gt = tf.squeeze(gt, channel_axis)
            gt_fg = tf.equal(tf.cast(gt, tf.float32), foreground_class_index)
            output_classes = tf.argmax(net_output, axis=channel_axis)
            output_foreground = tf.equal(output_classes, foreground_class_index)
            seg_measure = tf.py_function(seg_numpy, inp=[gt_fg, output_foreground], Tout=[tf.float32])
            return seg_measure[0]

    return calc_seg


def SegMeasure_unit_test():
    h = w = d = 30
    batch_size = 3

    # FOR UNIT TEST, Define random ground truth objects
    gt = np.zeros((batch_size, h, w, 1)).astype(np.uint16)

    objects = [(12, 20, 0, 5), (0, 9, 0, 5), (12, 20, 9, 20), (0, 9, 9, 20)]

    # FOR UNIT TEST, Define random output objects
    output = np.zeros((batch_size, h, w, 3)).astype(np.float32)
    output[:, :, :, 0] = 0.25

    i = 10
    for b in range(batch_size):

        for obj_id, (xs, xe, ys, ye) in enumerate(objects):
            gt[b, ys + i:ye + i, xs + i:xe + i] = obj_id + 1
            output[b, max(ys, 0):max(ye, 0), max(xs + i, 0):max(xe + i, 0), 1] = 0.5
        i += 1

    # The following two lines should be in your code:
    # Define the measure object:
    seg_measure = SegMeasure()
    # THE NEXT LINE CALLS THE CALCULATION
    measure_value = seg_measure(gt, output).numpy()
    print(measure_value)


if __name__ == "__main__":
    SegMeasure_unit_test()
