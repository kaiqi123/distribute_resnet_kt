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

    all_labels = []

    self.good_policies = found_policies.good_policies()

    # Determine how many data batched to load
    num_data_batches_to_load = 5
    total_batches_to_load = num_data_batches_to_load
    train_batches_to_load = total_batches_to_load
    assert hparams.train_size + hparams.validation_size <= 50000
    if hparams.eval_test:
      total_batches_to_load += 1
    # Determine how many images we have loaded
    total_dataset_size = 10000 * num_data_batches_to_load
    train_dataset_size = total_dataset_size
    if hparams.eval_test:
      total_dataset_size += 10000

    if hparams.dataset == 'cifar10':
      all_data = np.empty((total_batches_to_load, 10000, 3072), dtype=np.uint8)
    elif hparams.dataset == 'cifar100':
      assert num_data_batches_to_load == 5
      all_data = np.empty((1, 50000, 3072), dtype=np.uint8)
      if hparams.eval_test:
        test_data = np.empty((1, 10000, 3072), dtype=np.uint8)
    if hparams.dataset == 'cifar10':
      tf.logging.info('Cifar10')
      datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
      datafiles = datafiles[:train_batches_to_load]
      if hparams.eval_test:
        datafiles.append('test_batch')
      num_classes = 10
    elif hparams.dataset == 'cifar100':
      datafiles = ['train']
      if hparams.eval_test:
        datafiles.append('test')
      num_classes = 100
    else:
      raise NotImplementedError('Unimplemented dataset: ', hparams.dataset)
    if hparams.dataset != 'test':
      for file_num, f in enumerate(datafiles):
        d = unpickle(os.path.join(hparams.data_path, f))
        if f == 'test':
          test_data[0] = copy.deepcopy(d['data'])
          all_data = np.concatenate([all_data, test_data], axis=1)
        else:
          all_data[file_num] = copy.deepcopy(d['data'])
        if hparams.dataset == 'cifar10':
          labels = np.array(d['labels'])
        else:
          labels = np.array(d['fine_labels'])
        nsamples = len(labels)
        for idx in range(nsamples):
          all_labels.append(labels[idx])

    all_data = all_data.reshape(total_dataset_size, 3072)
    all_data = all_data.reshape(-1, 3, 32, 32)
    all_data = all_data.transpose(0, 2, 3, 1).copy()
    all_data = all_data / 255.0
    mean = augmentation_transforms.MEANS
    std = augmentation_transforms.STDS
    tf.logging.info('mean:{}    std: {}'.format(mean, std))
    all_data = (all_data - mean) / std

    #all_data, all_labels, train_dataset_size = selfself.read_random_data(all_data, all_labels, train_dataset_size, self.hparams)
    #all_data, all_labels, train_dataset_size = self.get_subset_categoriesReduced(all_data, all_labels, train_dataset_size, self.hparams)
    #all_data, all_labels, train_dataset_size = self.get_subset_categoriesReduced_TestDataDivideByLabels(all_data, all_labels, train_dataset_size, self.hparams)

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

  def get_subset_categoriesReduced(self, data, labels, train_dataset_size, hparams):
    tf.logging.info("Select sub data set categoriesReduced..................................")
    assert len(data) == len(labels)
    tf.logging.info("Before reducing categories, the number of data: {}".format(len(data)))

    train_data =  data[:train_dataset_size]
    test_data = data[train_dataset_size:]
    train_labels =  labels[:train_dataset_size]
    test_labels = labels[train_dataset_size:]
    tf.logging.info("Before reducing categories, the number of training data: {}".format(len(train_data)))
    tf.logging.info("Before reducing categories, the number of testing data: {}".format(len(test_data)))

    target_labels = [7,8,9]
    tf.logging.info("Select data with label: {}".format(target_labels))
    # select 0-6 categories from train dataset
    sub_data = []
    sub_labels = []
    for i in range(len(train_data)):
      if train_labels[i] in target_labels:
        sub_data.append(train_data[i])
        sub_labels.append(train_labels[i])
    assert len(sub_data) == len(sub_labels)
    sub_train_dataset_size = len(sub_data)

    # divide test data and labels into two parts, saw and unsaw
    test_data_saw = []
    test_labels_saw = []
    test_data_unsaw = []
    test_labels_unsaw = []
    for j in range(len(test_labels)):
      if test_labels[j] in target_labels:
        test_data_saw.append(test_data[j])
        test_labels_saw.append(test_labels[j])
      else:
        test_data_unsaw.append(test_data[j])
        test_labels_unsaw.append(test_labels[j])

    #tf.logging.info("test_labels_saw: {}".format(test_labels_saw))
    #tf.logging.info("test_labels_unsaw: {}".format(test_labels_unsaw))
    tf.logging.info("The number of all test data and test label: {}, {}".format(len(test_data), len(test_labels)))
    tf.logging.info("The number of seen test data and test label: {}, {}".format(len(test_data_saw), len(test_labels_saw)))
    tf.logging.info("The number of unseen test data and test label: {}, {}".format(len(test_data_unsaw), len(test_labels_unsaw)))

    # add test dataset
    if hparams.test_size == 10000:
      for j in range(len(test_labels)):
        sub_data.append(test_data[j])
        sub_labels.append(test_labels[j])
    elif hparams.test_size == 3000:
      assert hparams.test_size == len(test_labels_saw)
      for j in range(len(test_labels_saw)):
        sub_data.append(test_data_saw[j])
        sub_labels.append(test_labels_saw[j])
    elif hparams.test_size == 7000:
      assert hparams.test_size == len(test_labels_unsaw)
      for j in range(len(test_labels_unsaw)):
        sub_data.append(test_data_unsaw[j])
        sub_labels.append(test_labels_unsaw[j])
    else:
      raise ValueError("Not found test size!")

    sub_data = np.array(sub_data)
    assert len(sub_data) == len(sub_labels)
    assert len(sub_data) == sub_train_dataset_size+hparams.test_size
    tf.logging.info("After reducing categories, the number of data: {}".format(len(sub_data)))
    tf.logging.info("After reducing categories, the number of training data: {}".format(sub_train_dataset_size))
    tf.logging.info("After reducing categories, the number of testing data: {}".format(len(sub_data)-sub_train_dataset_size))
    return sub_data, sub_labels, sub_train_dataset_size

  def get_subset_categoriesReduced_TestDataDivideByLabels(self, data, labels, train_dataset_size, hparams):
    tf.logging.info("Select sub data set categoriesReduced..................................")
    assert len(data) == len(labels)
    tf.logging.info("Before reducing categories, the number of data: {}".format(len(data)))

    train_data =  data[:train_dataset_size]
    test_data = data[train_dataset_size:]
    train_labels =  labels[:train_dataset_size]
    test_labels = labels[train_dataset_size:]
    tf.logging.info("Before reducing categories, the number of training data: {}".format(len(train_data)))
    tf.logging.info("Before reducing categories, the number of testing data: {}".format(len(test_data)))

    target_labels = [3,4,8,9]
    unTarget_labels = [0,1,5,7]
    tf.logging.info("Select data with label: {}".format(target_labels))
    # select 0-6 categories from train dataset
    sub_data = []
    sub_labels = []
    for i in range(len(train_data)):
      if train_labels[i] in target_labels:
        sub_data.append(train_data[i])
        sub_labels.append(train_labels[i])
    assert len(sub_data) == len(sub_labels)
    sub_train_dataset_size = len(sub_data)

    # divide test data and labels into two parts, saw and unsaw
    test_data_saw = []
    test_labels_saw = []
    test_data_unsaw = []
    test_labels_unsaw = []
    for j in range(len(test_labels)):
      if test_labels[j] in target_labels:
        test_data_saw.append(test_data[j])
        test_labels_saw.append(test_labels[j])
      elif test_labels[j] in unTarget_labels:
        test_data_unsaw.append(test_data[j])
        test_labels_unsaw.append(test_labels[j])

    #tf.logging.info("test_labels_saw: {}".format(test_labels_saw))
    #tf.logging.info("test_labels_unsaw: {}".format(test_labels_unsaw))
    tf.logging.info("The number of all test data and test label: {}, {}".format(len(test_data), len(test_labels)))
    tf.logging.info("The number of seen test data and test label: {}, {}".format(len(test_data_saw), len(test_labels_saw)))
    tf.logging.info("The number of unseen test data and test label: {}, {}".format(len(test_data_unsaw), len(test_labels_unsaw)))

    # add test dataset
    if hparams.test_size == 10000:
      for j in range(len(test_labels)):
        sub_data.append(test_data[j])
        sub_labels.append(test_labels[j])
    elif hparams.test_size == 4000 and hparams.test_data_type == "Saw":
      tf.logging.info("Add saw test data")
      assert hparams.test_size == len(test_labels_saw)
      for j in range(len(test_labels_saw)):
        sub_data.append(test_data_saw[j])
        sub_labels.append(test_labels_saw[j])
    elif hparams.test_size == 4000 and hparams.test_data_type == "UnSaw":
      tf.logging.info("Add unsaw test data")
      assert hparams.test_size == len(test_labels_unsaw)
      for j in range(len(test_labels_unsaw)):
        sub_data.append(test_data_unsaw[j])
        sub_labels.append(test_labels_unsaw[j])
    #else:
    #  raise ValueError("Not found test size!")

    sub_data = np.array(sub_data)
    assert len(sub_data) == len(sub_labels)
    assert len(sub_data) == sub_train_dataset_size+hparams.test_size
    tf.logging.info("After reducing categories, the number of data: {}".format(len(sub_data)))
    tf.logging.info("After reducing categories, the number of training data: {}".format(sub_train_dataset_size))
    tf.logging.info("After reducing categories, the number of testing data: {}".format(len(sub_data)-sub_train_dataset_size))

    train_label_check = []
    test_label_check = []
    for label in sub_labels[:sub_train_dataset_size]:
      if label not in train_label_check:
        train_label_check.append(label)
    for label in sub_labels[sub_train_dataset_size:]:
      if label not in test_label_check:
        test_label_check.append(label)
    tf.logging.info("After reducing categories, the labels of training data include: {}".format(train_label_check))
    tf.logging.info("After reducing categories, the labels of testing data include: {}".format(test_label_check))

    return sub_data, sub_labels, sub_train_dataset_size

  def read_random_data(self, data, labels, train_dataset_size, hparams):
    tf.logging.info("read_random_data..................................")
    assert len(data) == len(labels)

    train_data =  data[:train_dataset_size]
    test_data = data[train_dataset_size:]
    train_labels =  labels[:train_dataset_size]
    test_labels = labels[train_dataset_size:]
    tf.logging.info("Before creating categories randomly, the number of training data: {}".format(len(train_data)))
    tf.logging.info("Before creating categories randomly, the number of testing data: {}".format(len(test_data)))

    # create train data and labels randomly, 5000
    # print(len(train_data), train_data[0].shape)
    train_data_random = np.random.random((5000, 32, 32, 3))
    train_labels_random = np.random.randint(0,10,size=[5000])
    sub_data = list(train_data_random)
    sub_labels = list(train_labels_random)
    assert len(sub_data) == len(sub_labels) == 5000
    sub_train_dataset_size = len(sub_data)
    #print(len(sub_data))

    # divide test data and labels into two parts, saw and unsaw
    target_labels = [0]
    tf.logging.info("Select data with label: {}".format(target_labels))

    test_data_saw = []
    test_labels_saw = []
    test_data_unsaw = []
    test_labels_unsaw = []
    for j in range(len(test_labels)):
      if test_labels[j] in target_labels:
        test_data_saw.append(test_data[j])
        test_labels_saw.append(test_labels[j])
      else:
        test_data_unsaw.append(test_data[j])
        test_labels_unsaw.append(test_labels[j])

    tf.logging.info("The number of all test data and test label: {}, {}".format(len(test_data), len(test_labels)))
    tf.logging.info("The number of seen test data and test label: {}, {}".format(len(test_data_saw), len(test_labels_saw)))
    tf.logging.info("The number of unseen test data and test label: {}, {}".format(len(test_data_unsaw), len(test_labels_unsaw)))

    # add unsaw test data and labels
    if hparams.test_size == 9000:
      assert hparams.test_size == len(test_labels_unsaw)
      for j in range(len(test_labels_unsaw)):
        sub_data.append(test_data_unsaw[j])
        sub_labels.append(test_labels_unsaw[j])
    else:
      raise ValueError("Not found test size!")

    sub_data = np.array(sub_data)
    assert len(sub_data) == len(sub_labels)
    assert len(sub_data) == sub_train_dataset_size+hparams.test_size
    return sub_data, sub_labels, sub_train_dataset_size

  # def next_batch(self):
  #   """Return the next minibatch of augmented data."""
  #   next_train_index = self.curr_train_index + self.hparams.batch_size
  #   if next_train_index > self.num_train:
  #     # Increase epoch number
  #     epoch = self.epochs + 1
  #     self.reset()
  #     self.epochs = epoch
  #   batched_data = (self.train_images[self.curr_train_index: self.curr_train_index + self.hparams.batch_size],
  #                   self.train_labels[self.curr_train_index: self.curr_train_index + self.hparams.batch_size])
  #   final_imgs = []
  #
  #   images, labels = batched_data
  #   for data in images:
  #     epoch_policy = self.good_policies[np.random.choice(len(self.good_policies))]
  #     final_img = augmentation_transforms.apply_policy(epoch_policy, data)
  #     final_img = augmentation_transforms.random_flip(augmentation_transforms.zero_pad_and_crop(final_img, 4))
  #     # Apply cutout
  #     final_img = augmentation_transforms.cutout_numpy(final_img)
  #     final_imgs.append(final_img)
  #   batched_data = (np.array(final_imgs, np.float32), labels)
  #   self.curr_train_index += self.hparams.batch_size
  #   return batched_data

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
