from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import custom_ops as ops
import numpy as np
import tensorflow as tf

student_output_dict = {}


def residual_block_student(x, in_filter, out_filter, stride, group_num, activate_before_residual=False, block_num=None):
  if activate_before_residual:
    with tf.variable_scope('shared_activation'):
      x = ops.batch_norm(x, scope='init_bn')
      x = tf.nn.relu(x)
      student_output_dict["group" + str(group_num) + "_block"+str(block_num)+"_sub1_relu"] = x
      orig_x = x
  else:
    orig_x = x

  block_x = x
  if not activate_before_residual:
    with tf.variable_scope('residual_only_activation'):
      block_x = ops.batch_norm(block_x, scope='init_bn')
      block_x = tf.nn.relu(block_x)
      student_output_dict["group" + str(group_num) + "_block"+str(block_num)+"_sub1_relu"] = block_x

  with tf.variable_scope('sub1'):
    block_x = ops.conv2d(block_x, out_filter, 3, stride=stride, scope='conv1')

  with tf.variable_scope('sub2'):
    block_x = ops.batch_norm(block_x, scope='bn2')
    block_x = tf.nn.relu(block_x)
    student_output_dict["group" + str(group_num) + "_block" + str(block_num) + "_sub2_relu"] = block_x
    block_x = ops.conv2d(block_x, out_filter, 3, stride=1, scope='conv2')

  with tf.variable_scope('sub_add'):  # If number of filters do not agree then zero pad them
    if in_filter != out_filter:
      orig_x = ops.avg_pool(orig_x, stride, stride)
      orig_x = ops.zero_pad(orig_x, in_filter, out_filter)

  x = orig_x + block_x
  return x


def _res_add(in_filter, out_filter, stride, x, orig_x):
  if in_filter != out_filter:
    #tf.logging.info("lst res_add, x: {}".format(x))
    #tf.logging.info("lst res_add, orig_x: {}".format(orig_x))
    orig_x = ops.avg_pool(orig_x, stride, stride)
    orig_x = ops.zero_pad(orig_x, in_filter, out_filter)
  x = x + orig_x
  orig_x = x
  return x, orig_x


def build_wrn_model_student(images, num_classes, wrn_size, num_blocks_per_resnet=4):

  with tf.variable_scope("student_architecture"):
      kernel_size = wrn_size
      filters = [min(kernel_size, 16), kernel_size, kernel_size * 2, kernel_size * 4]
      tf.logging.info('build_wrn_model_student, depth is {}, wide is {} '.format(6*num_blocks_per_resnet+4, wrn_size/16))

      filter_size = 3
      strides = [1, 2, 2]

      # Run the first conv
      with tf.variable_scope('init'):
        x = images
        output_filters = filters[0]
        x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')
        # student_output_dict["conv1"] = x
        tf.logging.info(str(x))

      first_x = x  # Res from the beginning
      orig_x = x  # Res from previous block

      for group_num in range(1, 4):
        with tf.variable_scope('group_{}'.format(group_num)):
          with tf.variable_scope('unit_{}_0'.format(group_num)):
            activate_before_residual = True if group_num == 1 else False
            x = residual_block_student(x, filters[group_num - 1], filters[group_num], strides[group_num - 1], group_num, activate_before_residual=activate_before_residual, block_num=0)
            tf.logging.info(str(x))
          for i in range(1, num_blocks_per_resnet):
            with tf.variable_scope('unit_{}_{}'.format(group_num, i)):
              x = residual_block_student(x, filters[group_num], filters[group_num], 1, group_num, activate_before_residual=False, block_num=i)
              tf.logging.info(str(x))
          with tf.variable_scope('res_add_{}'.format(group_num)):
            x, orig_x = _res_add(filters[group_num - 1], filters[group_num], strides[group_num - 1], x, orig_x)
          student_output_dict["group" + str(group_num) + "_out"] = x

      with tf.variable_scope('res_add_last'):
        final_stride_val = np.prod(strides)
        x, _ = _res_add(filters[0], filters[3], final_stride_val, x, first_x)
        tf.logging.info(str(x))

      with tf.variable_scope('unit_last'):
        x = ops.batch_norm(x, scope='final_bn')
        x = tf.nn.relu(x)
        student_output_dict["unit_last_relu"] = x
        x = ops.global_avg_pool(x)
        student_output_dict["global_avg_pool"] = x
        logits = ops.fc(x, num_classes)
        student_output_dict["fc"] = logits
        tf.logging.info(str(logits))
      return logits, student_output_dict