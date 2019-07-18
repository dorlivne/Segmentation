from model import TilesUnet, TilesUnetMirrored, Unet
from Loader import Loader
from loss import loss_fn
import tensorflow as tf
from configs import config


if __name__ == '__main__':
    Net_OOP = TilesUnet()
    config.mirror_threshold = 0.8
    config.use_assaf = True
    Net_OOP.compile(optimizer=tf.compat.v2.optimizers.Adam(beta_1=0.99), loss=loss_fn)
    Net_OOP.load_weights(filepath='/home/dorliv/Desktop/DNN/Segmentation/experiements/tiles_aug_saved_models/checkpoint.hdf5')
    loader = Loader(batch_size=1)
    accuracy, loss = Net_OOP.evaluation(loader=loader, epochs=100, imshow=False)
    print("accuracy {}, loss {}".format(accuracy, loss))