from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from Augmentations import IMAGE_HEIGHT,IMAGE_WIDTH
import tensorflow as tf

"""

def unet(pretrained_weights=None, input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer= 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3, 1)(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model
"""

def conv2D_BN_Relu(inputs, channel_Num, filter_size):
    conv = Conv2D(channel_Num, filter_size, padding='same', kernel_initializer='he_normal')(inputs)
    B1 = BatchNormalization()(conv)
    return ReLU()(B1)

"""
def unet(pretrained_weights=None, input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
    inputs = Input(input_size)
    conv1 = conv2D_BN_Relu(inputs, channel_Num=64, filter_size=3)
    conv1 = conv2D_BN_Relu(conv1, channel_Num=64, filter_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2D_BN_Relu(pool1, channel_Num=128, filter_size=3)
    conv2 = conv2D_BN_Relu(conv2, channel_Num=128, filter_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2D_BN_Relu(pool2, channel_Num=256, filter_size=3)
    conv3 = conv2D_BN_Relu(conv3, channel_Num=256, filter_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2D_BN_Relu(pool3, channel_Num=512, filter_size=3)
    conv4 = conv2D_BN_Relu(conv4, channel_Num=512, filter_size=3)
    #drop3 = Dropout(0.5)(conv3)

    up5 = conv2D_BN_Relu(UpSampling2D(size=(2, 2))(conv4), channel_Num=256, filter_size=2)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = conv2D_BN_Relu(merge5, channel_Num=256, filter_size=3)
    conv5 = conv2D_BN_Relu(conv5, channel_Num=256, filter_size=3)

    up6 = conv2D_BN_Relu(UpSampling2D(size=(2, 2))(conv5), channel_Num=128, filter_size=2)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = conv2D_BN_Relu(merge6, channel_Num=128, filter_size=3)
    conv6 = conv2D_BN_Relu(conv6, channel_Num=128, filter_size=3)

    up7 = conv2D_BN_Relu(UpSampling2D(size=(2, 2))(conv6), channel_Num=64, filter_size=2)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = conv2D_BN_Relu(merge7, channel_Num=64, filter_size=3)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv9 = Conv2D(3, 1,)(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

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
