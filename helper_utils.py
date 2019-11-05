from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import json
import helper_initialization
import helper_output_analyze
import time

def setup_loss(logits, labels):
  """Returns the cross entropy for the given `logits` and `labels`."""
  predictions = tf.nn.softmax(logits)
  cost = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
  return predictions, cost


def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost

def eval_child_model(session, model, data_loader, mode, summary_eval_writer=None):
  if mode == 'val':
    images = data_loader.val_images
    labels = data_loader.val_labels
  elif mode == 'test':
    images = data_loader.test_images
    labels = data_loader.test_labels
  elif mode == 'eval_train':
    images = data_loader.train_images
    labels = data_loader.train_labels
  else:
    raise ValueError('Not valid eval mode')
  assert len(images) == len(labels)

  # assert len(images) % model.batch_size == 0
  eval_batches = int(len(images) / model.batch_size)
  for i in range(eval_batches):
    eval_images = images[i * model.batch_size:(i + 1) * model.batch_size]
    eval_labels = labels[i * model.batch_size:(i + 1) * model.batch_size]
    _ = session.run(model.eval_op,
        feed_dict={
            model.images: eval_images,
            model.labels: eval_labels,
        })
    #summary_eval_writer.add_summary(summary, i)
  accuracy = session.run(model.accuracy)
  tf.logging.info('Evaluating mode is {}   accuracy is {}'.format(mode, accuracy))
  tf.logging.info('model.batch_size is {}'.format(model.batch_size))
  tf.logging.info('length of images is {}'.format(len(images)))
  return accuracy


def cosine_lr(learning_rate, epoch, iteration, batches_per_epoch, total_epochs):
  t_total = total_epochs * batches_per_epoch
  t_cur = float(epoch * batches_per_epoch + iteration)
  return 0.5 * learning_rate * (1 + np.cos(np.pi * t_cur / t_total))


def get_lr(curr_epoch, hparams, iteration=None):
  # resnet learning rate, decay with iterations
  assert iteration is not None
  batches_per_epoch = int(hparams.train_size / hparams.batch_size)
  lr = cosine_lr(hparams.lr, curr_epoch, iteration, batches_per_epoch, hparams.num_epochs)

  """
  # resnet learning rate, decay at hparams.lr_decay_epoch
  lr = hparams.lr
  lr_decay_epoch = json.loads(hparams.lr_decay_epoch)
  if curr_epoch >= int(lr_decay_epoch[0]):
    lr = hparams.lr * 0.2
  if curr_epoch >= int(lr_decay_epoch[1]):
    lr = hparams.lr * 0.2 * 0.2
  if curr_epoch >= int(lr_decay_epoch[2]):
    lr = hparams.lr * 0.2 * 0.2 * 0.2
  """
  return lr

def operations_for_KD(curr_epoch, step, session, model):
  #if curr_epoch == 0 and step == 0:
  #  model_dir = os.path.join(model.hparams.KD_checkpoint_dir,'model')
  #  checkpoint_path = tf.train.latest_checkpoint(model_dir)
  #  model.saverKD.restore(session, checkpoint_path)
  lamma = model.hparams.lamma_KD_initial - 0.015 * (curr_epoch + 1)
  model.lamma_KD_ph.load(lamma, session=session)
  if step == 0:
    tf.logging.info('lamma_KD of {} for epoch {}'.format(lamma, curr_epoch))

def compute_cosine_similarity(model, session, train_images, train_labels):
    cosine_lists, maxCosine_counts = session.run([model.cosine_lists, model.maxCosine_counts],
                                       feed_dict={
                                         model.images: train_images,
                                         model.labels: train_labels,
                                         model.teacher_model.images: train_images,
                                         model.teacher_model.labels: train_labels,
                                       })
    for i in range(len(cosine_lists)):
      count_cosine_lists[i].append(maxCosine_counts[i])
      max_index = cosine_lists[i].index(np.max(cosine_lists[i]))
      assert max_index == maxCosine_counts[i]
      # tf.logging.info("max_index: {}, maxCosine_counts: {}, cosine_lists: {}".format(max_index, maxCosine_counts[i], cosine_lists[i]))
    # np.save("output/count_cosine_lists.npy", count_cosine_lists)

    teacherlayers, teacher_group1_block3_sub1_relu, \
    teacher_group3_block0_sub1_relu, teacher_unit_last_relu = session.run(
      [model.teacherlayers, model.teacher_group1_block3_sub1_relu,
       model.teacher_group3_block0_sub1_relu, model.teacher_unit_last_relu],
      feed_dict={
        model.images: train_images,
        model.labels: train_labels,
        model.teacher_model.images: train_images,
        model.teacher_model.labels: train_labels,
      })
    print(teacherlayers[0][0][0][0][2], teacher_group1_block3_sub1_relu[0][0][0][2])
    print(teacherlayers[1][0][0][0][2], teacher_group3_block0_sub1_relu[0][0][0][2])
    print(teacherlayers[2][0][0][0][2], teacher_unit_last_relu[0][0][0][2])
    assert teacherlayers[0][0][0][0][2] == teacher_group1_block3_sub1_relu[0][0][0][2]
    assert teacherlayers[1][0][0][0][2] == teacher_group3_block0_sub1_relu[0][0][0][2]
    assert teacherlayers[2][0][0][0][2] == teacher_unit_last_relu[0][0][0][2]


def restore_variables_from_DeCAF_phase1(model, session):
    model_dir_DeCAF = os.path.join(model.hparams.DeCAF_checkpoint_dir, 'model')
    checkpoint_path_DeCAF = tf.train.latest_checkpoint(model_dir_DeCAF)
    model.saverDeCAF.restore(session, checkpoint_path_DeCAF)


student_avg_num0filters_dict_toatalEpochs = {"group1_block0_sub1_relu":[], "group1_block0_sub2_relu":[],
               "group2_block0_sub1_relu":[], "group2_block0_sub2_relu":[],
               "group3_block0_sub1_relu":[], "group3_block0_sub2_relu":[],
               "unit_last_relu":[], "fc": []}

count_cosine_lists = [[],[],[]]
def run_epoch_training(session, model, data_loader, curr_epoch):

  # Get the current learning rate for the model based on the current epoch
  curr_lr = get_lr(curr_epoch, model.hparams, iteration=0)
  tf.logging.info('lr of {} for epoch {}'.format(curr_lr, curr_epoch))

  steps_per_epoch = int(model.hparams.train_size / model.hparams.batch_size)
  steps_per_epoch = 10
  tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
  curr_step = session.run(model.global_step)
  print(curr_step)
  #assert curr_step % steps_per_epoch == 0

  # init avg_num0filters_dict_perEpoch, key the same as student_avg_num0filters_dict_toatalEpochs, value is []
  # if model.type == "independent_student":
  #     avg_num0filters_dict_perEpoch = {}
  #     for key in student_avg_num0filters_dict_toatalEpochs.keys():
  #         avg_num0filters_dict_perEpoch[key] = []

  step = curr_step
  while step < curr_step+steps_per_epoch:

    if step % 1 == 0:
        tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))

    train_images, train_labels = data_loader.next_batch()

    curr_lr = get_lr(curr_epoch, model.hparams, iteration=(step + 1))
    model.lr_rate_ph.load(curr_lr, session=session)

    if model.type == "dependent_student":

      # for cosine similarity
      #compute_cosine_similarity(model, session, train_images, train_labels)

      # for visualizing the output, not add normally, affecting the results
      #visualize_the_output_of_teacher(model, session, data_loader, step)

      # init
      #if curr_epoch == 0 and step==0:
      #  helper_initialization.initialize_conv_weights(session)
      #  helper_initialization.initialize_lastUnit_weights(session)
      #  helper_initialization.initialize_group123_gammaBeta(session)
      #  helper_initialization.initialize_lastUnit_gammaBeta(session)

      # KD
      #operations_for_KD(curr_epoch, step, session, model)

      # DeCAF, restore variables
      #if curr_epoch == 0 and step == 0:
      #    restore_variables_from_DeCAF_phase1(model, session)

      _, step, eval_op, teacher_eval_op, summary = session.run(
        [model.train_op, model.global_step, model.eval_op,
         model.teacher_model.eval_op, model.summary_op],
          feed_dict={
            model.images: train_images,
            model.labels: train_labels,
            model.teacher_model.images: train_images,
            model.teacher_model.labels: train_labels,
          })
    elif model.type == "independent_student" or model.type == "teacher":

      #if curr_epoch == 0 and step == 0:
      #    restore_variables_from_DeCAF_phase1(model, session)

      _, step, eval_op, summary = session.run(
        [model.train_op, model.global_step, model.eval_op, model.summary_op],
          feed_dict={
            model.images: train_images,
            model.labels: train_labels,
          })

      # analyze output of every layer for pruning
      # avg_num0filters_dict_perEpoch = helper_output_analyze.run_output_list_perIteration(session, model, train_images, train_labels, avg_num0filters_dict_perEpoch)

    else:
      raise EOFError("Not found model.type when training!")
    #summary_writer.add_summary(summary, step)

  train_accuracy = session.run(model.accuracy)
  tf.logging.info('Training accuracy: {}'.format(train_accuracy))

  if model.type == "dependent_student":
    teacher_accuracy = session.run(model.teacher_model.accuracy)
    tf.logging.info('Teacher accuracy: {}'.format(teacher_accuracy))

  #tf.logging.info('number of trainable params: {}'.format(model.num_trainable_params))

  # analyze output of every layer for pruning
  # if model.type == "independent_student" or model.type == "teacher":
  #     if model.type == "independent_student":
  #         avg_num0filters_dict_toatalEpochs = student_avg_num0filters_dict_toatalEpochs
  #
  #     for key, avg_num0filter_everyIteration_perEpoch in sorted(avg_num0filters_dict_perEpoch.items()):
  #         if key in avg_num0filters_dict_toatalEpochs.keys():
  #             avg_num0filters_dict_toatalEpochs[key].append(np.mean(avg_num0filter_everyIteration_perEpoch))
  #
  #     save_fileName = "./output_num0filter/"+str(model.hparams.checkpoint_dir).split("/")[2]+".txt"
  #     f = open(save_fileName,'w')
  #     f.write("Average number of 0 filter activations per epoch\n")
  #     f.write("Epoch: {}\n".format(curr_epoch))
  #     for key, value in sorted(avg_num0filters_dict_toatalEpochs.items()):
  #         assert len(avg_num0filters_dict_toatalEpochs[key]) == curr_epoch + 1
  #         f.write("Layer name: {}\n".format(key))
  #         f.write(str(value)+"\n")
  #         tf.logging.info(key)
  #         tf.logging.info(value)
  #     f.close()
  #     tf.logging.info("Save num0filter to {}".format(save_fileName))

  return train_accuracy
