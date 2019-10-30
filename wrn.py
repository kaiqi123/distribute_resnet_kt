from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import custom_ops as ops
import numpy as np
import tensorflow as tf

teacher_output_dict = {}
student_output_dict = {}


def residual_block_teacher(x, in_filter, out_filter, stride, group_num, activate_before_residual=False, block_num=None):
  if activate_before_residual:
    with tf.variable_scope('shared_activation'):
      x = ops.batch_norm(x, scope='init_bn')
      x = tf.nn.relu(x)
      teacher_output_dict["group" + str(group_num) + "_block"+str(block_num)+"_sub1_relu"] = x
      orig_x = x
  else:
    orig_x = x

  block_x = x
  if not activate_before_residual:
    with tf.variable_scope('residual_only_activation'):
      block_x = ops.batch_norm(block_x, scope='init_bn')
      block_x = tf.nn.relu(block_x)
      teacher_output_dict["group" + str(group_num) + "_block"+str(block_num)+"_sub1_relu"] = block_x

  with tf.variable_scope('sub1'):
    block_x = ops.conv2d(block_x, out_filter, 3, stride=stride, scope='conv1')

  with tf.variable_scope('sub2'):
    block_x = ops.batch_norm(block_x, scope='bn2')
    block_x = tf.nn.relu(block_x)
    teacher_output_dict["group" + str(group_num) + "_block" + str(block_num) + "_sub2_relu"] = block_x
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


def build_wrn_model_teacher(images, num_classes, wrn_size, num_blocks_per_resnet=4):

  with tf.variable_scope("teacher_architecture"):
      kernel_size = wrn_size
      filters = [min(kernel_size, 16), kernel_size, kernel_size * 2, kernel_size * 4]
      tf.logging.info('build_wrn_model_teacher, depth is {}, wide is {} '.format(6*num_blocks_per_resnet+4, wrn_size/16))

      filter_size = 3
      strides = [1, 2, 2]

      # Run the first conv
      with tf.variable_scope('init'):
        x = images
        output_filters = filters[0]
        x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')
        # teacher_output_dict["conv1"] = x
        tf.logging.info(str(x))

      first_x = x  # Res from the beginning
      orig_x = x  # Res from previous block

      for group_num in range(1, 4):
        with tf.variable_scope('group_{}'.format(group_num)):
          with tf.variable_scope('unit_{}_0'.format(group_num)):
            activate_before_residual = True if group_num == 1 else False
            x = residual_block_teacher(x, filters[group_num - 1], filters[group_num], strides[group_num - 1], group_num, activate_before_residual=activate_before_residual, block_num=0)
            tf.logging.info(str(x))
          for i in range(1, num_blocks_per_resnet):
            with tf.variable_scope('unit_{}_{}'.format(group_num, i)):
              x = residual_block_teacher(x, filters[group_num], filters[group_num], 1, group_num, activate_before_residual=False, block_num=i)
              tf.logging.info(str(x))
          with tf.variable_scope('res_add_{}'.format(group_num)):
            x, orig_x = _res_add(filters[group_num - 1], filters[group_num], strides[group_num - 1], x, orig_x)
          teacher_output_dict["group" + str(group_num) + "_out"] = x

      with tf.variable_scope('res_add_last'):
        final_stride_val = np.prod(strides)
        x, _ = _res_add(filters[0], filters[3], final_stride_val, x, first_x)
        tf.logging.info(str(x))

      with tf.variable_scope('unit_last'):
        x = ops.batch_norm(x, scope='final_bn')
        x = tf.nn.relu(x)
        teacher_output_dict["unit_last_relu"] = x
        x = ops.global_avg_pool(x)
        teacher_output_dict["global_avg_pool"] = x
        logits = ops.fc(x, num_classes)
        teacher_output_dict["fc"] = logits
        tf.logging.info(str(logits))
      return logits, teacher_output_dict


def residual_block_student(x, in_filter, out_filter, stride, group_num, activate_before_residual=False, block_num=0, num_convs_per_block=2):
  #tf.logging.info(out_filter)
  if activate_before_residual:  # Pass up RELU and BN activation for resnet
    with tf.variable_scope('shared_activation'):
      x = ops.batch_norm(x, scope='init_bn')
      x = tf.nn.relu(x)
      student_output_dict["group" + str(group_num) + "_block" + str(block_num) + "_sub1_relu"] = x
      orig_x = x
  else:
    orig_x = x

  block_x = x
  if not activate_before_residual:
    with tf.variable_scope('residual_only_activation'):
      block_x = ops.batch_norm(block_x, scope='init_bn')
      block_x = tf.nn.relu(block_x)
      student_output_dict["group" + str(group_num) + "_block" + str(block_num) + "_sub1_relu"] = block_x

  with tf.variable_scope('sub1'):
    block_x = ops.conv2d(block_x, out_filter[0], 3, stride=stride, scope='conv1')

  if num_convs_per_block == 2:
      with tf.variable_scope('sub2'):
        block_x = ops.batch_norm(block_x, scope='bn2')
        block_x = tf.nn.relu(block_x)
        student_output_dict["group" + str(group_num) + "_block" + str(block_num) + "_sub2_relu"] = block_x
        block_x = ops.conv2d(block_x, out_filter[1], 3, stride=1, scope='conv2')

      with tf.variable_scope('sub_add'):
        if in_filter != out_filter[1]:
          orig_x = ops.avg_pool(orig_x, stride, stride)
          orig_x = ops.zero_pad(orig_x, in_filter, out_filter[1])

  elif num_convs_per_block == 1:
      with tf.variable_scope('sub_add'):
        if in_filter != out_filter[0]:
          orig_x = ops.avg_pool(orig_x, stride, stride)
          orig_x = ops.zero_pad(orig_x, in_filter, out_filter[0])
  else:
      raise ValueError("Not found num_convs_per_block")

  x = orig_x + block_x

  return x

def build_wrn_model_independentStudent(images, num_classes, wrn_size, num_group, num_convs_per_block = 2, teacher_model=None):

  with tf.variable_scope("student_architecture"):

      kernel_size = wrn_size
      filters = [[16, 16], [160, 160], [320, 320], [640, 640]] # original
      #filters = [[16-2, 16-2], [160-40, 160-22], [320-34, 320-48], [640-546, 640]] # 100% 0
      #filters = [[16-4, 16-4], [160-64, 160-28], [320-94, 320-54], [640-580, 640-18]] # 90% 0
      #filters = [[16-4, 16-4], [160-92, 160-50], [320-216, 320-74], [640-616, 640-58]] # 80% 0
      #filters = [[16-4, 16-4], [160-106, 160-82], [320-272, 320-110], [640-632, 640-142]] # 70% 0
      #filters = [[16-6, 16-6], [160-118, 160-130], [320-304, 320-216], [640-638, 640-340]] # 50% 0

      tf.logging.info('build_wrn_model_independentStudent, depth is {}, wide is {} '.format((num_group-1) * num_convs_per_block + 4, wrn_size / 16))

      filter_size = 3
      strides = [1, 2, 2]  # stride for each resblock

      # Run the first conv
      with tf.variable_scope('init'):
        x = images
        output_filters = filters[0][0]
        x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')
        tf.logging.info(str(x))

      first_x = x  # Res from the beginning
      orig_x = x  # Res from previous block

      for group_num in range(1, num_group):
        with tf.variable_scope('group_{}'.format(group_num)):
            with tf.variable_scope('unit_{}_0'.format(group_num)):
              activate_before_residual = True if group_num == 1 else False
              x = residual_block_student(x, filters[group_num - 1][1], filters[group_num], strides[group_num - 1], group_num, activate_before_residual=activate_before_residual, block_num=0, num_convs_per_block=num_convs_per_block)

            # with tf.variable_scope('res_add_{}'.format(group_num)):
            #     if num_convs_per_block == 2:
            #         x, orig_x = _res_add(filters[group_num - 1][1], filters[group_num][1], strides[group_num - 1], x, orig_x)
            #     elif num_convs_per_block == 1:
            #         x, orig_x = _res_add(filters[group_num - 1][0], filters[group_num][0], strides[group_num - 1], x, orig_x)
            #     else:
            #         raise ValueError("Not found num_convs_per_block")
            tf.logging.info(str(x))

      with tf.variable_scope('res_add_last'):
          final_stride_val = np.prod(strides[0:num_group - 1])

          if num_convs_per_block == 2:
              x, _ = _res_add(filters[0][0], filters[num_group - 1][1], final_stride_val, x, first_x)
          elif num_convs_per_block == 1:
              x, _ = _res_add(filters[0][0], filters[num_group - 1][0], final_stride_val, x, first_x)
          else:
              raise ValueError("Not found num_convs_per_block")
          tf.logging.info(str(x))

      with tf.variable_scope('unit_last'):
        x = ops.batch_norm(x, scope='final_bn')
        x = tf.nn.relu(x)
        student_output_dict["unit_last_relu"] = x
        x = ops.global_avg_pool(x)
        logits = ops.fc(x, num_classes)
        student_output_dict["fc"] = logits
        tf.logging.info(str(x))
        tf.logging.info(str(logits))
      return logits, student_output_dict


"""
if teacher_model is not None:
  x = teacher_model.output_dict["group1_out"] - x

with tf.variable_scope('change_dimension'):
def change_dimension(in_filter, out_filter, stride, orig_x):
    orig_x = ops.avg_pool(orig_x, stride, stride)
    orig_x = ops.zero_pad(orig_x, in_filter, out_filter)
    return orig_x
x = change_dimension(160, 640, 4, x)
tf.logging.info(x)
"""