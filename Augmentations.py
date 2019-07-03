import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
import matplotlib.pyplot as plt
from configs import config

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 640


def flip_randomly(image, seg):
    prob = random.random()
    if random.random() > prob:
        image = np.fliplr(image)
        seg = np.fliplr(seg)
    if random.random() > prob:
        image = np.flipud(image)
        seg = np.flipud(seg)
    return image, seg


def jitter_image(train_batch, train_seg_batch):
        height = IMAGE_HEIGHT
        width = IMAGE_WIDTH
        PLOT = False
        jitter_images = np.zeros((train_batch.shape[0], height, width, 1))
        jitter_seg = np.zeros((train_batch.shape[0], height, width, 1))
        for i, image in enumerate(train_batch):
                seg = train_seg_batch[i]
                if PLOT:
                    imshow_noax(image.squeeze())
                    imshow_noax(seg.squeeze())
                if random.random() < config.elastic_threshold:
                    transformed_image, transformed_seg = elastic_transformation(image=image.squeeze(), seg=seg.squeeze())
                else:
                    transformed_image = image.squeeze()
                    transformed_seg = seg.squeeze()
                if PLOT:
                    imshow_noax(transformed_image)
                    imshow_noax(transformed_seg)
                distorted_image, distorted_seg = flip_randomly(transformed_image, transformed_seg)
                if PLOT:
                    imshow_noax(distorted_image)
                    imshow_noax(distorted_seg)
                jitter_images[i] = np.expand_dims(distorted_image,-1)
                jitter_seg[i] = np.expand_dims(distorted_seg,-1)
        return jitter_images.astype(np.float32), jitter_seg.astype(np.float32)


def elastic_transformation(image, seg, alpha=np.random.random(1)*10, sigma=4, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.
         """
        assert len(image.shape) == 2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        transformed_image = map_coordinates(image, indices, order=1).reshape(shape)
        transformed_seg = map_coordinates(seg, indices, order=1).reshape(shape)
        return transformed_image, transformed_seg


def imshow_noax(img, normalize=False):
    """ Tiny helper to show images remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 55.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img)
    plt.gca().axis('off')
    plt.show()
