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
    self.good_policies = found_policies.good_policies()
    self.seed = 0

    if hparams.dataset == 'cifar10':
      train_dataset_path = ""
      test_dataset_path = ""
      self.image_size = 32
    else:
      raise NotImplementedError('Unimplemented dataset: ', hparams.dataset)

    train_size = self.check_data(file_name=train_dataset_path)
    self.train_image, self.train_label = self.read_one_image_label(dataset_path = train_dataset_path)

    test_size = self.check_data(file_name=test_dataset_path)
    test_image, test_label = self.read_one_image_label(dataset_path = test_dataset_path)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * test_size
    self.test_images, self.test_labels = tf.train.shuffle_batch(
      [test_image, test_label], batch_size=test_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue, seed=self.seed)

  def check_data(self, file_name):
    with open(file_name) as f:
      lines = f.readlines()
    labels = []
    for line in lines:
      label = int(line.split(",")[0])
      if label not in labels:
        labels.append(label)
    data_size = len(lines)
    print("Read data from file: {}".format(file_name))
    print("The number of data is: {}".format(data_size))
    print("The labels are: {}".format(sorted(labels)))
    return data_size

  def read_one_image_label(self, dataset_path):
    filename_queue = tf.train.string_input_producer([dataset_path], num_epochs=None)
    _, value_temp = tf.TextLineReader().read(filename_queue)
    record_defaults = [[1], ['']]
    label, image_path = tf.decode_csv(value_temp, record_defaults=record_defaults)
    file_content = tf.read_file(image_path)
    image = tf.image.decode_jpeg(file_content, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, [self.image_size, self.image_size])
    return image, label

  def next_batch(self, num_gpus):
    batch_size_total = self.hparams.batch_size * num_gpus
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size_total
    self.train_images_batch, self.train_labels_batch = tf.train.shuffle_batch(
      [self.train_image, self.train_label], batch_size=batch_size_total, capacity=capacity,
      min_after_dequeue=min_after_dequeue, seed=self.seed)
    return self.train_images_batch, self.train_labels_batch
