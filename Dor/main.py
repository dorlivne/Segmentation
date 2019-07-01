from Loader import Loader
from model import unet
from loss import loss_fn
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import callbacks
from configs import config
import numpy as np
import os
from loss import create_weights
from multiprocessing import Process, Queue


def extract_weight_map(queue_seg: Queue, queue_weightmap: Queue):
    seg = queue_seg.get()
    queue_weightmap.put(create_weights(seg=seg))
    extract_weight_map(queue_seg, queue_weightmap)


def validation(net: Model, loader: Loader):
    val_accuracy = []
    loss = []
    queue_seg = Queue(maxsize=1)
    queue_weightmap = Queue(maxsize=1)
    p = Process(target=extract_weight_map, args=(queue_seg, queue_weightmap))
    p.start()
    for i, (image_batch, seg_batch) in enumerate(loader.get_minibatch(train=False)):
        queue_seg.put(seg_batch)
        pred = net(image_batch)
        weight_map = queue_weightmap.get()
        loss.append(loss_fn(seg=seg_batch, predictions=pred, weight_map=weight_map))
        val_accuracy.append(Jaccard_Index(predictions=pred, seg=seg_batch))
        if i == config.validation_steps:
            return np.mean(val_accuracy), np.mean(loss)
    p.terminate()


# TODO implement this function
def Jaccard_Index(predictions, seg):
    return 0


def train(net: Model, optimizer: tf.compat.v2.optimizers, loader: Loader):
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    queue_seg = Queue(maxsize=1)
    queue_weightmap = Queue(maxsize=1)
    p = Process(target=extract_weight_map, args=(queue_seg, queue_weightmap))
    p.start()
    Tensorcallback = callbacks.TensorBoard(config.log_dir,
                                           write_graph=False, write_images=False)
    Tensorcallback.set_model(net)
    for epoch in range(config.epochs):
        train_accuracy = []
        train_loss = []
        for iteration, (image_batch, seg_batch) in enumerate(loader.get_minibatch()):
            queue_seg.put(seg_batch)
            with tf.GradientTape() as tape:
                pred = net(image_batch)
                weight_map = queue_weightmap.get()
                train_loss.append(loss_fn(seg_batch, pred, weight_map=weight_map))
                train_accuracy.append(Jaccard_Index(pred, seg_batch))
                print("Epoch {}/{} iteration {} : loss {} accuracy {}".
                      format(epoch, config.epochs, iteration, train_loss[-1], train_accuracy[-1]))
            grads = tape.gradient(train_loss[-1], net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            if iteration == config.steps_per_epoch:
                break
        train_loss_mean = np.mean(train_loss)
        train_accuracy_mean = np.mean(train_accuracy)
        if epoch % config.val_freq == 0:  # output validation accuracy
            val_accuracy, val_loss = validation(net, loader)
            logs = {'acc': train_accuracy_mean, 'loss': train_loss_mean, 'val_acc': val_accuracy, 'val_loss': val_loss}
            Tensorcallback.on_epoch_end(epoch=epoch, logs=logs)
    Tensorcallback.on_train_end('_')
    p.terminate()


        # TODO write results to logger and print them, if val converges then stop procedure
        # TODO add learning rate scheduler


if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    loader = Loader(batch_size=1)
    Net = unet()
    optimizer = tf.compat.v2.optimizers.Adam()
    train(Net, optimizer, loader)

