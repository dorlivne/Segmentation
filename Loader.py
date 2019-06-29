import os
import tensorflow as tf
from Augmentations import jitter_image


def create_dataset(csv_filename):
    image_paths = tf.data.experimental.CsvDataset(filenames=csv_filename, record_defaults=[tf.string, tf.string])
    data_set_raw = []
    data_set_seg = []
    for raw_path, seg_path in image_paths:
        raw_path = str(raw_path.numpy(), 'utf-8')
        raw_path = os.path.join(root_dir, raw_path[2:])
        seg_path = str(seg_path.numpy(), 'utf-8')
        seg_path = os.path.join(root_dir, seg_path[2:])
        raw_data = tf.io.read_file(filename=raw_path)
        seg_data = tf.io.read_file(filename=seg_path)
        raw_image = tf.image.decode_image(raw_data, dtype=tf.uint16)
        seg_image = tf.image.decode_image(seg_data, dtype=tf.uint8)
        data_set_raw.append(raw_image)
        data_set_seg.append(seg_image)
    ans = tf.data.Dataset.from_tensor_slices((data_set_raw, data_set_seg))
    return ans


class Loader():
    """
    Loader class is Used to load train\validation batches of Raw images and Seg images.
    Loader class has the possibility to augment the data.
    """
    def __init__(self, root_dir=r'Data', batch_size=16):
        train_csv = os.path.join(root_dir, 'train.csv')
        val_csv = os.path.join(root_dir, 'val.csv')
        self.train_dataset = create_dataset(train_csv)
        self.val_dataset = create_dataset(val_csv)
        self.train_dataset = self.train_dataset.repeat(None)  # Repeat dataset indefinetly
        self.val_dataset = self.val_dataset.repeat(None)  # Repeat dataset indefinetly
        self.train_dataset = self._get_train_batch(batch_size=batch_size)
        self.val_dataset = self._get_val_batch(batch_size=batch_size)

    def _get_train_batch(self, batch_size=2):
        return self.train_dataset.batch(batch_size=batch_size)

    def _get_val_batch(self, batch_size=2):
        return self.val_dataset.batch(batch_size=batch_size)

    def get_minibatch(self, train=True):
        if train:
            multiple_batches = self.train_dataset
        else:
            multiple_batches = self.val_dataset
        for image_batch, seg_batch in multiple_batches:
            yield jitter_image(train_batch=image_batch.numpy(), train_seg_batch=seg_batch.numpy())


if __name__ == '__main__':
    root_dir = r'Data'
    loader = Loader(root_dir=root_dir)
    for batch_raw, batch_seg in loader.get_minibatch():
        a = 5
