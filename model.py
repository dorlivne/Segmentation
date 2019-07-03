from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from Augmentations import IMAGE_HEIGHT, IMAGE_WIDTH, imshow_noax
import tensorflow as tf
from utils.custom_utils import crop, Jaccard_Index, np, crop_and_mirror
from Loader import Loader
from utils.logger_utils import info
from configs import config
import random


def conv2D_BN_Relu(inputs, channel_Num, filter_size):
    conv = Conv2D(channel_Num, filter_size, padding='same', kernel_initializer='he_normal')(inputs)
    B1 = BatchNormalization()(conv)
    return ReLU()(B1)

"""
def unet(pretrained_weights=None, input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
    inputs = Input(input_size)
    conv1 = conv2D_BN_Relu(inputs, channel_Num=32, filter_size=3)
    conv1 = conv2D_BN_Relu(conv1, channel_Num=32, filter_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2D_BN_Relu(pool1, channel_Num=64, filter_size=3)
    conv2 = conv2D_BN_Relu(conv2, channel_Num=64, filter_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2D_BN_Relu(pool2, channel_Num=128, filter_size=3)
    conv3 = conv2D_BN_Relu(conv3, channel_Num=128, filter_size=3)
    #drop3 = Dropout(0.5)(conv3)

    up4 = conv2D_BN_Relu(UpSampling2D(size=(2, 2))(conv3), channel_Num=64, filter_size=2)
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = conv2D_BN_Relu(merge4, channel_Num=64, filter_size=3)
    conv4 = conv2D_BN_Relu(conv4, channel_Num=64, filter_size=3)

    up5 = conv2D_BN_Relu(UpSampling2D(size=(2, 2))(conv4), channel_Num=32, filter_size=2)
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = conv2D_BN_Relu(merge5, channel_Num=32, filter_size=3)
    conv5 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv6 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(3, 1,)(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
"""


def unet(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
    """

    :param input_size: the inputs size to the network
    :return: a Keras model module
    """
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer= 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    up4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv3))
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up5 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv6 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(3, 1, )(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    return model


class Unet(tf.keras.Model):
    """
    Unet class for segmentation tasks
    """

    def __init__(self, input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
        super(Unet, self).__init__()
        self.model = unet(input_size=input_size)  # building the net architecture from unet function

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: image batches that serve as an input to the net
        :param training: flag for training mode, not relevent for this module
        :return:
        """
        pred = self.model(inputs)
        return pred

    def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
        # defines the  optimizer and the loss function for the model
        self.optimizer = optimizer
        self.loss = loss

    def _train_on_batch(self, image_batch, seg_batch):
        """
        private function to train on batches
        :param image_batch: image batch of raw segmentation
        :param seg_batch:  seg batch of ground truth segmentations
        :return:  accuracy and loss mean
        """
        image_batch = image_batch.astype(np.float32)
        seg_batch = seg_batch.astype(np.float32)
        with tf.GradientTape() as tape:
            pred = self(image_batch)
            train_loss = self.loss(seg_batch, pred)
            train_accuracy = Jaccard_Index(output_batch=pred, gt_batch=seg_batch, visual=False)
        grads = tape.gradient(train_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return train_accuracy, train_loss

    def fit(self, logger, loader: Loader, callbacks, epochs:int, steps_per_epoch:int,
            val_freq=2, val_steps=4):
        """
        training loop that trains the model and controls the augmentation, learning rate and the mirroring capability.
        the training loops stops on two conditions:
            A. the val accuracy converges over N consecutive trials
            B. ran for "epochs" number of epochs

        :param logger: logger to log information
        :param loader: generator of data
        :param callbacks: a dict of tensorboard callbacks and checkpoint callbacks
        :param epochs: number of epochs to run
        :param steps_per_epoch: number of iterations per epoch
        :param val_freq: frequency of epochs to preform validation
        :param val_steps: number of steps to preform in validation
        """
        Tensorcallback = callbacks['tensorboard']
        Checkpoint = callbacks['checkpoint']
        info("----- Start Train -----", logger)
        best_val_acc = 0
        consecutive_trials_val_acc_unchanged = 0
        for epoch in range(epochs):
            if best_val_acc > 0.7:   # a threshold for Curriculum learning
                config.elastic_threshold = 0.5  # for the use of elastic threshold
            if best_val_acc > 0.6:
                config.mirror_threshold = 0.5  # for the use of mirroring
            train_accuracy = []
            train_loss = []
            self.optimizer.learning_rate = config.learning_rate_scheduler(epoch)  # learning rate scheduler
            for iteration, (image_batch, seg_batch) in enumerate(loader.get_minibatch(Aug=config.Augment)):
                acc_val, loss_val = self._train_on_batch(image_batch=image_batch, seg_batch=seg_batch)
                train_accuracy.append(acc_val)
                train_loss.append(loss_val)
                if iteration == steps_per_epoch:
                    break
            train_loss_mean = np.mean(train_loss)
            train_accuracy_mean = np.mean(train_accuracy)
            if epoch % val_freq == 0:  # output validation accuracy
                val_accuracy, val_loss = self.validation(loader=loader)
                info("--- Epoch {}/{}: train accuracy"
                     " {}, val accuracy {} and loss {} with learning rate {}--- "
                     .format(epoch, epochs, train_accuracy_mean, val_accuracy, train_loss_mean,
                             self.optimizer.learning_rate.numpy()), logger)
                logs = {'acc': train_accuracy_mean, 'loss': train_loss_mean, 'val_acc': val_accuracy,
                        'val_loss': val_loss}
                Tensorcallback.on_epoch_end(epoch=epoch, logs=logs)
                Checkpoint.on_epoch_end(epoch=epoch, logs=logs)
                if best_val_acc < val_accuracy:
                    best_val_acc = val_accuracy
                    consecutive_trials_val_acc_unchanged = 0
                    info("-- Epoch {}/{} : best val so far is {} ---".format(epoch, epochs, best_val_acc),
                         logger)
                else:
                    consecutive_trials_val_acc_unchanged += 1
            if consecutive_trials_val_acc_unchanged >= config.converged_threshold:  # check if val converged
                info("--- Epoch {}/{} : val acc converged for {} consecutive trials, terminating training... ---"
                     .format(epoch, epochs, config.converged_threshold), logger)
                break
        Tensorcallback.on_train_end('_')

    def validation(self, loader: Loader):
            """
            validation function that computes the Jaccard Index accuracy on the val data set
            :param loader: generator of data
            :return: mean of accuracy and loss
            """
            val_accuracy = []
            loss = []
            for i, (image_batch, seg_batch) in enumerate(loader.get_minibatch(train=False)):
                pred = self.call(image_batch.astype(np.float32), training=False)
                loss.append(self.loss(seg=seg_batch.astype(np.float32), predictions=pred))
                val_accuracy.append(Jaccard_Index(output_batch=pred, gt_batch=seg_batch, visual=False))
                if i == config.validation_steps:
                    return np.mean(val_accuracy), np.mean(loss)

    def evaluation(self, loader: Loader, epochs:int, imshow=True):
        # for debugging purposes
        val_accuracy = []
        loss = []
        for i, (image_batch, seg_batch) in enumerate(loader.get_minibatch(train=False)):
            pred = self.call(image_batch.astype(np.float32), training=False)
            loss.append(self.loss(seg=seg_batch.astype(np.float32), predictions=pred))
            val_accuracy.append(Jaccard_Index(output_batch=pred, gt_batch=seg_batch, visual=imshow))
            if i == epochs:
                return np.mean(val_accuracy), np.mean(loss)


class TilesUnet(Unet):
    """
    TilesUnet module, based on the original Unet where the inputs are tiles of the original image.
    """
    def __init__(self, input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), number_of_tiles=4):
        super(TilesUnet, self).__init__()
        self.input_size = input_size
        self.divide_by = int(np.sqrt(number_of_tiles))
        # we divide two dims so we need to square root the number of tiles
        self.model = unet(input_size=(input_size[0]/self.divide_by, input_size[1]/self.divide_by, 1))

    def call(self, inputs, training=True, mask=None):
        """
        overide the call function of Unet
        :param inputs: batches of raw images
        :param training: when training is True we compare an image tile to a seg tile, when training is False
                         we tile together the tiles to create a full prediction of the segmentation
        :return: prediction
        """
        if training:
            return super(TilesUnet, self).call(inputs)
        else:
            sub_image, sub_image_width, sub_image_height = crop(image_batch=inputs, return_offset=True)
            prediction = np.zeros((np.shape(inputs)[0], IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            for i in range(self.divide_by):  # height
                for j in range(self.divide_by):  # width
                    sub_prediction = self.model(sub_image[:, j + 2 * i, :, :]).numpy()
                    prediction[:, i * sub_image_height:(1+i) * sub_image_height,
                    j * sub_image_width: (j+1) * sub_image_width, :] = sub_prediction

            return prediction

    def _train_on_batch(self, image_batch, seg_batch):
        sub_image = crop(image_batch=image_batch) # crop function is used to divide the image to tiles
        sub_seg = crop(image_batch=seg_batch)
        train_acc = []
        train_lost = []
        for i in range(self.divide_by):  # height
                for j in range(self.divide_by):  # width
                    train_acc_val, train_loss_val = super(TilesUnet, self).\
                        _train_on_batch(image_batch=sub_image[:, j + 2 * i, :, :],
                                        seg_batch=sub_seg[:, j + 2 * i, :, :])
                    train_acc.append(train_acc_val)
                    train_lost.append(train_loss_val)
        return np.mean(train_acc), np.mean(train_lost)


class TilesUnetMirrored(Unet):
    """
    TilesUnetMirrored module is module with the same size of the Unet module, it has the capability to create tiles
    of the input images and mirror them in order to create more data that fits the scale of the original image
    """
    def __init__(self, number_of_tiles=4, input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
        super(TilesUnetMirrored, self).__init__()
        self.input_size = input_size
        self.divide_by = int(np.sqrt(number_of_tiles))

    def _train_on_batch(self, image_batch, seg_batch):
        mirror = False
        if random.random() < config.mirror_threshold:
            #  we randomly insert mirrored cropped images to the dataset
            mirrored_and_cropped_images = crop_and_mirror(image_batch=image_batch, num_of_sub_pictures=self.divide_by)
            mirrored_and_cropped_seg = crop_and_mirror(image_batch=seg_batch, num_of_sub_pictures=self.divide_by)
            mirror = True
        train_acc = []
        train_lost = []
        for i in range(self.divide_by):  # height
            for j in range(self.divide_by):  # width
                if mirror:
                    train_acc_val, train_loss_val = super(TilesUnetMirrored, self). \
                        _train_on_batch(image_batch=mirrored_and_cropped_images[:, j + 2 * i, :, :],
                                        seg_batch=mirrored_and_cropped_seg[:, j + 2 * i, :, :])
                else:
                    train_acc_val, train_loss_val = super(TilesUnetMirrored, self). \
                        _train_on_batch(image_batch=image_batch.astype(np.float32),
                                        seg_batch=seg_batch.astype(np.float32))
                train_acc.append(train_acc_val)
                train_lost.append(train_loss_val)
        return np.mean(train_acc), np.mean(train_lost)








