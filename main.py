from Loader import Loader
from model import Unet, TilesUnet, TilesUnetMirrored
from loss import loss_fn
import tensorflow as tf
from tensorflow.python.keras import callbacks
from configs import config
import os
from utils.logger_utils import get_logger
from argparse import ArgumentParser

FLAGS = 0


def info(string, logger):
    print(string)
    logger.info(string)

"""
def validation(net: Model, loader: Loader, epoch):
    val_accuracy = []
    loss = []
    #net.prob.assign(0.0)
    for i, (image_batch, seg_batch) in enumerate(loader.get_minibatch(train=False)):
        pred = net(image_batch)
        loss.append(loss_fn(seg=seg_batch, predictions=pred, epoch=epoch))
        val_accuracy.append(Jaccard_Index(output_batch=pred, gt_batch=seg_batch, visual=False))
        if i == config.validation_steps:
            #net.prob.assign(0.25)
            return np.mean(val_accuracy), np.mean(loss)

def train(net: Model, optimizer: tf.compat.v2.optimizers, loader: Loader):
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)
    logger = get_logger(os.path.join(FLAGS.log_dir, "train_log"))
    Tensorcallback = callbacks.TensorBoard(FLAGS.log_dir,
                                           write_graph=False, write_images=False)
    Checkpoint = callbacks.ModelCheckpoint(filepath=FLAGS.model_path + "/checkpoint.hdf5", monitor='val_acc', mode='max',
                                           save_best_only=True)
    Checkpoint.set_model(net)
    Tensorcallback.set_model(net)
    best_val_acc = 0
    consecutive_trials_val_acc_unchanged = 0
    info("----- Start Train -----", logger)
    for epoch in range(FLAGS.epochs):
        if epoch == 50 :
            config.elastic_threshold = 0.5
        train_accuracy = []
        train_loss = []
        for iteration, (image_batch, seg_batch) in enumerate(loader.get_minibatch(Aug=config.Augment)):
            with tf.GradientTape() as tape:
                pred = net(image_batch)
                train_loss.append(loss_fn(seg_batch, pred,epoch=epoch))
                train_accuracy.append(Jaccard_Index(output_batch=pred, gt_batch=seg_batch, visual=False))
            optimizer.learning_rate = config.learning_rate_scheduler(epoch)
            grads = tape.gradient(train_loss[-1], net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            if iteration == config.steps_per_epoch:
                break
        train_loss_mean = np.mean(train_loss)
        train_accuracy_mean = np.mean(train_accuracy)
        if epoch % config.val_freq == 0:  # output validation accuracy
            val_accuracy, val_loss = validation(net, loader, epoch)
            info("--- Epoch {}/{}: train accuracy"
                 " {}, val accuracy {} and loss {} with learning rate {}--- "
                 .format(epoch, FLAGS.epochs, train_accuracy_mean, val_accuracy, train_loss_mean,
                         optimizer.learning_rate.numpy()), logger)
            logs = {'acc': train_accuracy_mean, 'loss': train_loss_mean, 'val_acc': val_accuracy, 'val_loss': val_loss}
            Tensorcallback.on_epoch_end(epoch=epoch, logs=logs)
            Checkpoint.on_epoch_end(epoch=epoch, logs=logs)
            if best_val_acc < val_accuracy:
                best_val_acc = val_accuracy
                consecutive_trials_val_acc_unchanged = 0
                info("-- Epoch {}/{} : best val so far is {} ---".format(epoch, FLAGS.epochs, best_val_acc), logger)
            else:
                consecutive_trials_val_acc_unchanged += 1
        if consecutive_trials_val_acc_unchanged >= config.converged_threshold:
            info("--- Epoch {}/{} : val acc converged for {} consecutive epochs, terminating training... ---"
                 .format(epoch, FLAGS.epochs, config.converged_threshold), logger)
            break
    Tensorcallback.on_train_end('_')
"""


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
    parser.add_argument(
        '--log_dir',
        type=str,
        default=config.log_dir,
        help='Log Directory.')
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='pre trained model path.')
    FLAGS, unparsed = parser.parse_known_args()

    Net_OOP = TilesUnetMirrored()
    optimizer = tf.compat.v2.optimizers.Adam(beta_1=0.99)
    loader = Loader(batch_size=FLAGS.batch_size)
    Net_OOP.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc', 'loss', 'val_acc', 'val_loss'])
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)
    logger = get_logger(os.path.join(FLAGS.log_dir, "train_log"))
    Tensorcallback = callbacks.TensorBoard(FLAGS.log_dir,
                                           write_graph=False, write_images=False)
    Checkpoint = callbacks.ModelCheckpoint(filepath=FLAGS.model_path + "/checkpoint.hdf5", monitor='val_acc', mode='max',
                                           save_best_only=True)
    Checkpoint.set_model(Net_OOP)
    Tensorcallback.set_model(Net_OOP)
    callbacks = {'tensorboard': Tensorcallback, 'checkpoint': Checkpoint}
    Net_OOP.fit(logger=logger, callbacks=callbacks,epochs=FLAGS.epochs, steps_per_epoch=config.steps_per_epoch,
                val_freq=config.val_freq, val_steps=config.validation_steps, loader=loader)

