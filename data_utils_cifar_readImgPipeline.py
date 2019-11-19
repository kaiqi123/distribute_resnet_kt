from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import augmentation_transforms
import numpy as np
import policies as found_policies
import tensorflow as tf
try:
  import cPickle
except ImportError:
  import _pickle as cPickle

class DataSetCifar(object):

  def __init__(self, hparams):
    self.hparams = hparams
    self.epochs = 0
    self.curr_train_index = 0
    self.good_policies = found_policies.good_policies()

    if hparams.dataset == 'cifar10':
      tf.logging.info('Cifar10')
      datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5','test_batch']
      all_data, all_labels = self.read_pklData(hparams.data_path, datafiles)
      num_classes = 10
      train_dataset_size = hparams.train_size
      total_dataset_size = hparams.train_size + hparams.test_size
    else:
      raise NotImplementedError('Unimplemented dataset: ', hparams.dataset)

    all_data = all_data.reshape(total_dataset_size, 3072)
    all_data = all_data.reshape(-1, 3, 32, 32)
    all_data = all_data.transpose(0, 2, 3, 1).copy()
    all_data = all_data / 255.0
    mean = augmentation_transforms.MEANS
    std = augmentation_transforms.STDS
    tf.logging.info('mean:{}    std: {}'.format(mean, std))
    all_data = (all_data - mean) / std

    all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
    assert len(all_data) == len(all_labels)
    tf.logging.info('In CIFAR10 loader, number of images: {}'.format(len(all_data)))

    # Break off test data
    if hparams.eval_test:
      self.test_images = all_data[train_dataset_size:]
      self.test_labels = all_labels[train_dataset_size:]

    # Shuffle the rest of the data
    all_data = all_data[:train_dataset_size]
    all_labels = all_labels[:train_dataset_size]
    np.random.seed(0)
    perm = np.arange(len(all_data))
    np.random.shuffle(perm)
    all_data = all_data[perm]
    all_labels = all_labels[perm]

    # Break into train and val
    train_size, val_size = hparams.train_size, hparams.validation_size
    assert train_dataset_size >= train_size + val_size
    assert train_dataset_size == train_size
    self.train_images = all_data[:train_size]
    self.train_labels = all_labels[:train_size]
    self.val_images = all_data[train_size:train_size + val_size]
    self.val_labels = all_labels[train_size:train_size + val_size]
    self.num_train = self.train_images.shape[0]

  def read_pklData(self, data_path, datafiles):
      all_data = []
      all_labels = []
      for file_num, f in enumerate(datafiles):
          d = unpickle(os.path.join(data_path, f))
          data = d['data']
          labels = d['labels']
          all_data = all_data + list(data)
          all_labels = all_labels + labels
      all_data = np.array(all_data)
      return all_data, all_labels

  def next_batch(self, num_gpus):
    """Return the next minibatch of augmented data."""
    next_train_index = self.curr_train_index + self.hparams.batch_size*num_gpus
    if next_train_index > self.num_train:
      # Increase epoch number
      epoch = self.epochs + 1
      self.reset()
      self.epochs = epoch
    batched_data = (self.train_images[self.curr_train_index: self.curr_train_index + self.hparams.batch_size*num_gpus],
                    self.train_labels[self.curr_train_index: self.curr_train_index + self.hparams.batch_size*num_gpus])
    final_imgs = []

    images, labels = batched_data
    for data in images:
      epoch_policy = self.good_policies[np.random.choice(len(self.good_policies))]
      final_img = augmentation_transforms.apply_policy(epoch_policy, data)
      final_img = augmentation_transforms.random_flip(augmentation_transforms.zero_pad_and_crop(final_img, 4))
      # Apply cutout
      final_img = augmentation_transforms.cutout_numpy(final_img)
      final_imgs.append(final_img)
    batched_data = (np.array(final_imgs, np.float32), labels)
    self.curr_train_index += self.hparams.batch_size*num_gpus
    return batched_data


  def reset(self):
    """Reset training data and index into the training data."""
    self.epochs = 0
    # Shuffle the training data
    perm = np.arange(self.num_train)
    np.random.shuffle(perm)
    assert self.num_train == self.train_images.shape[0], 'Error incorrect shuffling mask'
    self.train_images = self.train_images[perm]
    self.train_labels = self.train_labels[perm]
    self.curr_train_index = 0

def unpickle(f):
  print('loading file: {}'.format(f))
  try:
    fo = tf.gfile.Open(f, 'r')
    d = cPickle.load(fo)
  except UnicodeDecodeError:
    fo = tf.gfile.Open(f, 'rb')
    d = cPickle.load(fo, encoding='latin1')
  fo.close()
  return d
