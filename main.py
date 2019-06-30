from Loader import Loader
from model import unet
from loss import loss_fn
import tensorflow as tf
from tensorflow.python.keras.models import Model
from configs import config
import time
# TODO implement this function
def validation_walkthrough(net: Model, loader: Loader):
    return 0

# TODO implement this function
def Jaccard_Index(predictions, seg):
    return 0


def train(net: Model, optimizer: tf.compat.v2.optimizers, loader: Loader):
    for epoch in range(config.epochs):
        accuracy_per_iteration = 0
        for iteration, (image_batch, seg_batch) in enumerate(loader.get_minibatch()):
            with tf.GradientTape() as tape:
                pred = net(image_batch)
                t = time.time()
                loss_value = loss_fn(seg_batch, pred)
                print("elapsed time {}".format(time.time() - t))
                accuracy_per_iteration = accuracy_per_iteration + Jaccard_Index(pred, seg_batch)
            epoch_accuracy = accuracy_per_iteration / 5

            grads = tape.gradient(loss_value, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            if iteration == 4:
                break
        if epoch % 2 == 0:  # output validation accuracy
            val_accuracy = validation_walkthrough(net, loader)
        print("epoch {} / {} :  loss value is {}".format(epoch, config.epochs, loss_value))
        # TODO write results to logger and print them, if val converges then stop procedure
        # TODO add learning rate scheduler


if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    loader = Loader(batch_size=8)
    Net = unet()
    optimizer = tf.compat.v2.optimizers.Adam()
    train(Net, optimizer, loader)

