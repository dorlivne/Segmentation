import tensorflow as tf
from Augmentations import IMAGE_HEIGHT, IMAGE_WIDTH, imshow_noax
import numpy as np
import matplotlib.pyplot as plt



# TODO implement this function
def Jaccard_Index(gt_batch, output_batch, channel_axis=-1, foreground_class_index=1, visual=False):
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
        plt.imshow(gt_batch[0])
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(output_classes[0])
        plt.colorbar()
        plt.show()

    return seg_measure


def is_whole(num):
    return num % 1 == 0


def crop(image_batch, num_of_sub_pictures=2, return_offset=False):
    """
    crop function to crop the image to tiles
    """
    sub_image_width = IMAGE_WIDTH / num_of_sub_pictures
    sub_image_height = IMAGE_HEIGHT / num_of_sub_pictures
    assert is_whole(sub_image_width)
    assert is_whole(sub_image_height)
    batch_size = np.shape(image_batch)[0]
    sub_image_width = int(sub_image_width)
    sub_image_height = int(sub_image_height)
    sub_images = np.zeros((batch_size, num_of_sub_pictures ** 2, sub_image_height, sub_image_width, 1))
    height_pixel_offset = 0
    width_pixel_offset = 0
    for i in range(num_of_sub_pictures):
        for j in range(num_of_sub_pictures):
            sub_images[:, j + 2 * i, :, :] = image_batch[:, height_pixel_offset: height_pixel_offset + sub_image_height,
                                          width_pixel_offset: width_pixel_offset + sub_image_width]
            width_pixel_offset = width_pixel_offset + sub_image_width
        height_pixel_offset = height_pixel_offset + sub_image_height

        width_pixel_offset = 0
    if return_offset:
        return sub_images.astype(np.float32), sub_image_width, sub_image_height
    else:
        return sub_images.astype(np.float32)


def crop_and_mirror(image_batch, num_of_sub_pictures=2):
    """
    crop the image to tile and then mirror the tiles to create a full scale image
    :param image_batch:
    :param num_of_sub_pictures:
    :return:
    """
    cropped_images, sub_image_width, sub_image_height = crop(image_batch, num_of_sub_pictures, True)
    #imshow_noax(cropped_images[0,0,:,:,0])
    mirror_images = batch_mirror_image(sub_image_width, sub_image_height ,cropped_images)
    return mirror_images


def batch_mirror_image(sub_image_width, sub_image_height, cropped_images, num_of_sub_pictures=2):
    mirrored_image = np.zeros((np.shape(cropped_images)[0], int(num_of_sub_pictures ** 2), IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    for i in range(num_of_sub_pictures):
        for j in range(num_of_sub_pictures):
            #imshow_noax(cropped_images[0, 0, :, :, 0])
            mirrored_image[:, i * 2 + j, 0:sub_image_height, 0:sub_image_width, 0] \
                = cropped_images[:, i * 2 + j, :, :, 0]
            #imshow_noax(mirrored_image[0, 0, :, :, 0])
            mirrored_image[:, i * 2 + j, 0:sub_image_height, sub_image_width : 2 * sub_image_width, 0] \
                = mirror_rows(cropped_images[:, i * 2 + j, :, :, 0])
            #imshow_noax(mirrored_image[0, 0, :, :, 0])
            mirrored_image[:, i * 2 + j, sub_image_height: 2 * sub_image_height, :, 0] \
                = mirror_columns(mirrored_image[:, i * 2 + j, 0:sub_image_height, :, 0])
            #imshow_noax(mirrored_image[0, 0, :, :, 0])
    return mirrored_image


def mirror_rows(img):
    height_size = np.shape(img)[2]
    mirrored_image = np.zeros_like(img)
    for column in range(height_size):
        mirrored_image[:, :, height_size - 1 - column] = img[:, :, column]
    return mirrored_image


def mirror_columns(img):
    row_size = np.shape(img)[1]
    mirrored_image = np.zeros_like(img)
    for row in range(row_size):
        mirrored_image[:, row_size - 1 - row, :] = img[:, row,:]
    return mirrored_image