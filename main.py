from Loader import Loader
from model import unet
from loss import loss_fn
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import callbacks
from configs import config
import numpy as np
import os
from multiprocessing import Process, Queue
from utils.custom_utils import extract_weight_map, Jaccard_Index
from utils.logger_utils import get_logger
from argparse import ArgumentParser
FLAGS = 0


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


def train(net: Model, optimizer: tf.compat.v2.optimizers, loader: Loader):
    logger = get_logger(os.path.join(config.log_dir, 'train_log'))
    queue_seg = Queue(maxsize=1)
    queue_weightmap = Queue(maxsize=1)
    p = Process(target=extract_weight_map, args=(queue_seg, queue_weightmap))
    p.start()
    Tensorcallback = callbacks.TensorBoard(config.log_dir,
                                           write_graph=False, write_images=False)
    Checkpoint = callbacks.ModelCheckpoint(filepath=FLAGS.model_path, monitor='val_acc', mode='max')
    Checkpoint.set_model(net)
    Tensorcallback.set_model(net)
    best_val_acc = 0
    consecutive_trials_val_acc_unchanged = 0
    logger.info("starting training procedure...")
    for epoch in range(FLAGS.epochs):
        train_accuracy = []
        train_loss = []
        for iteration, (image_batch, seg_batch) in enumerate(loader.get_minibatch()):
            queue_seg.put(seg_batch)
            with tf.GradientTape() as tape:
                pred = net(image_batch)
                weight_map = queue_weightmap.get()
                train_loss.append(loss_fn(seg_batch, pred, weight_map=weight_map))
                train_accuracy.append(Jaccard_Index(pred, seg_batch))
                logger.info("Epoch {}/{} iteration {} : loss {} accuracy {}".
                      format(epoch, FLAGS.epochs, iteration, train_loss[-1], train_accuracy[-1]))
            optimizer.learning_rate = config.learning_rate_schedulere(epoch)
            grads = tape.gradient(train_loss[-1], net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            if iteration == config.steps_per_epoch:
                break

        train_loss_mean = np.mean(train_loss)
        train_accuracy_mean = np.mean(train_accuracy)
        print("--- Epoch {}/{}: train accuracy {} and loss {} --- "
              .format(epoch, FLAGS.epochs, train_accuracy_mean, train_loss_mean))
        logger.info("--- Epoch {}/{}: train accuracy {} and loss {} --- "
              .format(epoch, FLAGS.epochs, train_accuracy_mean, train_loss_mean))
        if epoch % config.val_freq == 0:  # output validation accuracy
            val_accuracy, val_loss = validation(net, loader)
            logs = {'acc': train_accuracy_mean, 'loss': train_loss_mean, 'val_acc': val_accuracy, 'val_loss': val_loss}
            Tensorcallback.on_epoch_end(epoch=epoch, logs=logs)
            Checkpoint.on_epoch_end(epoch=epoch, logs=logs)
            if best_val_acc < val_accuracy:
                best_val_acc = val_accuracy
                consecutive_trials_val_acc_unchanged = 0
                logger.info("-- Epoch {}/{} : best val so far is {} ---".format(epoch, FLAGS.epochs, best_val_acc))
                print("-- Epoch {}/{} : best val so far is {} ---".format(epoch, FLAGS.epochs, best_val_acc))
            else:
                consecutive_trials_val_acc_unchanged += 1
                logger.info("--- Epoch {}/{} : val acc unchanged for {} consecutive trials with learning rate {} ---"
                            .format(epoch, FLAGS.epochs,
                                    consecutive_trials_val_acc_unchanged, optimizer.learning_rate))
        if consecutive_trials_val_acc_unchanged >= 5:
            print("--- Epoch {}/{} : val acc converged for 10 consecutive epochs, terminating training... ---"
                  .format(epoch, FLAGS.epochs))
            logger.info("--- Epoch {}/{} : val acc converged for 10 consecutive epochs, terminating training... ---"
                        .format(epoch, FLAGS.epochs))
            break

    Tensorcallback.on_train_end('_')
    p.terminate()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=config.model_path,
        help=' Directory where to save model checkpoint.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.epochs,
        help='number of epoches')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.batch_size,
        help='number of epoches')
    FLAGS, unparsed = parser.parse_known_args()
    Net = unet()
    train(Net, tf.compat.v2.optimizers.Adam(), Loader(batch_size=FLAGS.batch_size))

