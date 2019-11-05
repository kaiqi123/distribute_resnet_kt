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

def get_lr(hparams, t_cur=None):
  assert t_cur is not None
  batches_per_epoch = int(hparams.train_size / hparams.batch_size)
  t_total = hparams.num_epochs * batches_per_epoch
  t_cur = float(t_cur)
  lr = 0.5 * hparams.lr * (1 + np.cos(np.pi * t_cur / t_total))
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

def run_iteration_training(session, model, data_loader, curr_step, total_steps):
    train_images, train_labels = data_loader.next_batch()
    curr_lr = get_lr(hparams=model.hparams, t_cur=curr_step)
    model.lr_rate_ph.load(curr_lr, session=session)

    if curr_step % 20 == 0:
        tf.logging.info('Training {}/{}, lr {}'.format(curr_step, total_steps, curr_lr))

    if model.type == "dependent_student":
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
      _, step, eval_op, summary = session.run(
        [model.train_op, model.global_step, model.eval_op, model.summary_op],
          feed_dict={
            model.images: train_images,
            model.labels: train_labels,
          })
    else:
      raise EOFError("Not found model.type when training!")
    return step

def calculate_training_accuracy(session, model):
    training_accuracy = session.run(model.accuracy)
    tf.logging.info('Training accuracy: {}'.format(training_accuracy))

    if model.type == "dependent_student":
        teacher_accuracy = session.run(model.teacher_model.accuracy)
        tf.logging.info('Teacher accuracy: {}'.format(teacher_accuracy))
    return training_accuracy

def show_accuracy_list(session, curr_epoch, model_train, model_eval, data_loader, training_accuracy_list, train_accuracy_list, test_accuracy_list):
    if curr_epoch != 1:
        with open("accuracy/training_accuracy.json", 'r') as f:
            training_accuracy_list = json.load(f)

    training_accuracy = calculate_training_accuracy(session, model_train)
    train_accuracy = eval_child_model(session, model_eval, data_loader, 'eval_train')
    test_accuracy = eval_child_model(session, model_eval, data_loader, 'test')

    training_accuracy_list.append(float(training_accuracy))
    train_accuracy_list.append(float(train_accuracy))
    test_accuracy_list.append(float(test_accuracy))

    with open("accuracy/training_accuracy.json", 'w') as f:
        json.dump(training_accuracy_list, f)

    return training_accuracy_list
