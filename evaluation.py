from model import TilesUnet
from Loader import Loader
from loss import loss_fn
import tensorflow as tf

if __name__ == '__main__':
    Net_OOP = TilesUnet()
    Net_OOP.compile(optimizer=tf.compat.v2.optimizers.Adam(beta_1=0.99), loss=loss_fn)
    Net_OOP.load_weights(filepath='/home/dorliv/Desktop/DNN/Segmentation/tiles_no_aug_saved_models/checkpoint.hdf5')
    loader = Loader(batch_size=1)
    accuracy, loss = Net_OOP.evaluation(loader=loader, epochs=100, imshow=True)
    print("accuracy {}".format(accuracy))