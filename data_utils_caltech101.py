from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import cPickle
import os
import augmentation_transforms
import numpy as np
import policies as found_policies
import tensorflow as tf

IMAGE_SIZE = 32

class DataSetCaltech101(object):

  def __init__(self, hparams):
    self.hparams = hparams
    self.epochs = 0
    self.curr_train_index = 0
    self.good_policies = found_policies.good_policies()

    if hparams.dataset == 'caltech101':
        datafiles = ['train_1.pkl', 'train_2.pkl', 'train_3.pkl', 'train_4.pkl', 'train_5.pkl', 'test.pkl']
        all_data, all_labels = self.read_pklData("./caltech101-batches-py", datafiles)
        num_classes = 102
        train_dataset_size = 7280
    else:
      raise NotImplementedError('Unimplemented dataset: ', hparams.dataset)

    all_data = all_data.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    all_data = all_data.transpose(0, 2, 3, 1).copy()
    all_data = all_data / 255.0

    print(type(all_data), all_data.shape)
    print(type(all_labels), len(all_labels))

    mean = augmentation_transforms.MEANS
    std = augmentation_transforms.STDS
    tf.logging.info('mean:{}    std: {}'.format(mean, std))

    all_data = (all_data - mean) / std
    #all_data, all_labels, train_dataset_size = self.get_subset(all_data, all_labels, train_dataset_size)
    all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
    assert len(all_data) == len(all_labels)

    print(type(all_data), all_data.shape)
    print(type(all_labels), len(all_labels))

    tf.logging.info('In Caltech101 loader, number of images: {}'.format(len(all_data)))

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

  def get_subset(self, data, labels, train_dataset_size):
    print("Select sub data set..................................")
    #print(len(data))
    #print(train_dataset_size)
    #print(data[0:2])
    train_data =  data[:train_dataset_size]
    test_data = data[train_dataset_size:]
    train_labels =  labels[:train_dataset_size]
    test_labels = labels[train_dataset_size:]

    # select 0-6 categories from train dataset
    sub_data = []
    sub_labels = []
    for i in range(len(train_data)):
      if train_labels[i] >= 0 and train_labels[i] <= 6:
        sub_data.append(train_data[i])
        sub_labels.append(train_labels[i])
    assert len(sub_data) == len(sub_labels)
    sub_train_dataset_size = len(sub_data)

    # add original 10000 test dataset
    for j in range(len(test_labels)):
      sub_data.append(test_data[j])
      sub_labels.append(test_labels[j])
    #print(len(sub_data))
    #print(len(sub_labels))
    #print(sub_data[0:2])
    sub_data = np.array(sub_data)
    assert len(sub_data) == len(sub_labels)
    assert len(sub_data) == sub_train_dataset_size+len(test_labels)
    return sub_data, sub_labels, sub_train_dataset_size

  def next_batch(self):
    """Return the next minibatch of augmented data."""
    next_train_index = self.curr_train_index + self.hparams.batch_size
    if next_train_index > self.num_train:
      # Increase epoch number
      epoch = self.epochs + 1
      self.reset()
      self.epochs = epoch
    batched_data = (self.train_images[self.curr_train_index: self.curr_train_index + self.hparams.batch_size],
                    self.train_labels[self.curr_train_index: self.curr_train_index + self.hparams.batch_size])

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

    self.curr_train_index += self.hparams.batch_size
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
  tf.logging.info('loading file: {}'.format(f))
  fo = tf.gfile.Open(f, 'r')
  d = cPickle.load(fo)
  fo.close()
  return d
