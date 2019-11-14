from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
try:
    from contextlib import nested
except ImportError:
    from contextlib import ExitStack, contextmanager
    @contextmanager
    def nested(*contexts):
        with ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            yield contexts
# import logging
import os
import time
import json
import numpy as np
import tensorflow as tf

import custom_ops as ops
import data_utils_cifar
import data_utils_imagenet
import helper_buildLoss
import helper_utils
import helper_cosineSimilarity
import helper_buildLoss_comparedMethods
import helper_buildLoss_DeCAF
import wrn_student_comparedMethod
import helper_output_analyze
from wrn import build_wrn_model_independentStudent
from wrn import build_wrn_model_teacher

tf.flags.DEFINE_string('model_name', '#', 'the name of model')
tf.flags.DEFINE_string('checkpoint_dir', './#', 'Training Directory.')
tf.flags.DEFINE_string('data_path', './cifar-10-batches-py', 'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', 'cifar10', 'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_string('teacher_checkpoint_dir', './trained_teacher', 'Trained Teacher Directory.')
tf.flags.DEFINE_string('model_type', '#', 'the type of model')
tf.flags.DEFINE_integer('train_size', 50000, 'the size of training examples')
tf.flags.DEFINE_integer('test_size', 10000, 'the size of testing examples')
tf.flags.DEFINE_integer('batch_size', 128, 'the size of one batch of training examples')
#tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')
tf.flags.DEFINE_float('lamma_KD_initial', 4.0, 'lamma_KD_initial')
tf.flags.DEFINE_string('KD_checkpoint_dir', './trained_FitNet_phase1/#', 'KD_checkpoint_dir')
tf.flags.DEFINE_string('DeCAF_checkpoint_dir', '#', 'DeCAF_checkpoint_dir')
tf.flags.DEFINE_string('test_data_type', '#', 'test_data_type')
tf.flags.DEFINE_string('teacher_model_name', '#', 'the name of teacher model')

# For distributed
# tf.flags.DEFINE_string("ps_hosts", "#", "Comma-separated list of hostname:port pairs")
# tf.flags.DEFINE_string("worker_hosts", "#", "Comma-separated list of hostname:port pairs")
# tf.flags.DEFINE_string("job_name", "#", "One of 'ps', 'worker'")
# tf.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.flags.DEFINE_integer("num_gpus", 4, "num_gpus")

FLAGS = tf.flags.FLAGS
arg_scope = tf.contrib.framework.arg_scope

# logName = "./logs/"+str(FLAGS.checkpoint_dir).split("/")[2]+".log"
# if not os.path.exists(logName):
#     os.system(r"touch {}".format(logName))
# logging.basicConfig(filename=logName,level=logging.DEBUG)
# logging.FileHandler(logName, mode='w')


def setup_arg_scopes(is_training):
  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': True,
      'is_training': is_training,
  }
  scopes = []
  scopes.append(arg_scope([ops.batch_norm], **batch_norm_params))
  return scopes


def build_model(inputs, num_classes, is_training, hparams, type, teacher_model=None):
  scopes = setup_arg_scopes(is_training)
  with  nested(*scopes):
    if type == "teacher":
      if hparams.teacher_model_name == 'teacher_wrn_w1d40':
        logits, output_dict = build_wrn_model_teacher(inputs, num_classes, wrn_size=16*1, num_blocks_per_resnet=6)
      elif hparams.teacher_model_name == 'teacher_wrn_w2d40':
        logits, output_dict = build_wrn_model_teacher(inputs, num_classes, wrn_size=16*2, num_blocks_per_resnet=6)
      elif hparams.teacher_model_name == 'teacher_wrn_w10d28':
        logits, output_dict = build_wrn_model_teacher(inputs, num_classes, wrn_size=16*10, num_blocks_per_resnet=4)

    elif type == "independent_student":
      if hparams.model_name == 'independentStudent_wrn_w10d10':
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=4, num_convs_per_block=2)
      elif hparams.model_name == 'independentStudent_wrn_w10d7':  # block has one conv, 3+4=7
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=4, num_convs_per_block=1)
      elif hparams.model_name == 'independentStudent_wrn_w10d6':
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=2, num_convs_per_block=2)
      elif hparams.model_name == 'independentStudent_wrn_w10d5':  # block has one conv, 1+4=5
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=2, num_convs_per_block=1)

      # at, ft
      elif hparams.model_name == 'independentStudent_wrn_w1d16':
        logits, output_dict = wrn_student_comparedMethod.build_wrn_model_student(inputs, num_classes, wrn_size=16*1, num_blocks_per_resnet=2)
      elif hparams.model_name == 'independentStudent_wrn_w2d16':
        logits, output_dict = wrn_student_comparedMethod.build_wrn_model_student(inputs, num_classes, wrn_size=16*2, num_blocks_per_resnet=2)

    elif type == "dependent_student":
      if hparams.model_name == 'dependentStudent_wrn_w10d10':
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=4, num_convs_per_block=2, teacher_model=teacher_model)
      elif hparams.model_name == 'dependentStudent_wrn_w10d7':
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=4, num_convs_per_block=1, teacher_model=teacher_model)
      elif hparams.model_name == 'dependentStudent_wrn_w10d6':
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=2, num_convs_per_block=2, teacher_model=teacher_model)
      elif hparams.model_name == 'dependentStudent_wrn_w10d5': # block has one conv, 1+4=5
        logits, output_dict = build_wrn_model_independentStudent(inputs, num_classes, hparams.wrn_size, num_group=2, num_convs_per_block=1, teacher_model=teacher_model)

      elif hparams.model_name == 'dependentStudent_wrn_w1d16':
        logits, output_dict = wrn_student_comparedMethod.build_wrn_model_student(inputs, num_classes, wrn_size=16*1, num_blocks_per_resnet=2)

    else:
      raise ValueError("Not found model name")
  return logits, output_dict


class CifarModel(object):
  def __init__(self, hparams, type):
    self.hparams = hparams
    self.type = type

  def build(self, mode, images, labels, teacher_model=None):
    assert mode in ['train', 'eval']
    self.mode = mode
    self._setup_misc(mode)
    self._setup_images_and_labels(images, labels)

    if self.type in ["teacher", "independent_student"]:
      self._build_graph_independent(self.images, self.labels, mode)

    if self.type == "dependent_student":
      self.teacher_model = teacher_model
      self._build_graph_dependent_student(self.images, self.labels, mode)

      teacher_variables_to_restore = [var for var in tf.trainable_variables() if var.op.name.startswith("model/teacher_architecture")]
      self.teacher_saver = tf.train.Saver(teacher_variables_to_restore)

    # self.summary_op = self.summary_ops()
    # self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

  # def summary_ops(self):
  #     if self.type == "independent_student" or self.type == "teacher":
  #       tf.summary.scalar(self.type + "/accuracy", self.accuracy)
  #       tf.summary.scalar(self.type + "/loss", self.cost)
  #     elif self.type == "dependent_student" and self.mode == "train":
  #       tf.summary.scalar(self.type + "/accuracy", self.accuracy)
  #       tf.summary.scalar(self.type + "/loss_with_correctLabel", self.cost)
  #       tf.summary.scalar(self.teacher_model.type + "/accuracy", self.teacher_model.accuracy)
  #     summary_op = tf.summary.merge_all()
  #     return summary_op

  def _setup_misc(self, mode):
    self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
    self.reuse = None if (mode == 'train') else True
    # self.batch_size = self.hparams.batch_size

  # def _setup_images_and_labels(self, images, labels):
  #   if FLAGS.dataset == 'cifar10':
  #     self.num_classes = 10
  #     image_size = 32
  #   elif FLAGS.dataset == 'cifar100':
  #     self.num_classes = 100
  #     image_size = 32
  #   elif FLAGS.dataset == 'caltech101':
  #     self.num_classes = 102
  #     image_size = 32
  #   else:
  #     raise ValueError("Not found dataSet name")
  #   self.images = images
  #   self.labels = labels
  #   # self.images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, image_size, image_size, 3], name="images_placeholder")
  #   # self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_classes], name="labels_placeholder")

  def assign_epoch(self, session, epoch_value):
    session.run(self._epoch_update, feed_dict={self._new_epoch: epoch_value})

  def _calc_num_trainable_params(self):
    self.num_trainable_params = np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
    tf.logging.info('number of trainable params: {}'.format(self.num_trainable_params))

  def _build_graph_dependent_student(self, images, labels, mode):

    def build_loss_and_train_op(hparams, initial_lr, cost, teacher_output_dict, student_output_dict, global_step):
        for key, v in sorted(teacher_output_dict.items()):
          tf.logging.info('teacher_output_dict, key: {}, shape: {}'.format(key, v.shape))
        for key, v in sorted(student_output_dict.items()):
          tf.logging.info('student_output_dict, key: {}, shape: {}'.format(key, v.shape))

        optimizer = tf.train.MomentumOptimizer(initial_lr, 0.9, use_nesterov=True)
        tvars = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/")]
        grads = tf.gradients(cost, tvars)
        if hparams.gradient_clipping_by_global_norm > 0.0:
            grads, norm = tf.clip_by_global_norm(grads, hparams.gradient_clipping_by_global_norm)
        apply_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_step')
        for tvar in tvars:
          tf.logging.info('tvar: {}'.format(tvar))
        tf.logging.info('model/student_architecture/  trainable variables: {}'.format(len(tvars)))

        # DeCAF phase2
        #apply_op_list, loss_list = helper_buildLoss_DeCAF.build_train_op_DeCAF_onlyUpdateFc_usingTeacherFc(teacher_output_dict, student_output_dict, optimizer, tvars)

        # adding cosineSimilarity, 3reluFc
        #apply_op_list, teacherlayers, cosine_lists, maxCosine_counts = helper_cosineSimilarity.build_3reluFc_Loss_forD10_usingCosine(teacher_output_dict, student_output_dict, optimizer, tvars)

        # using Output Of Group1,2,3
        #apply_op_list, loss_list = helper_buildLoss_usingOtherApproaches.build_3OutputOfGroupFc_Loss_forD10(teacher_output_dict, student_output_dict, optimizer, tvars)

        # using Avg and Fc
        #apply_op_list, loss_list = helper_buildLoss_usingOtherApproaches.build_4fc_Loss_forD10(teacher_output_dict, student_output_dict, optimizer, tvars)

        # update all before layers
        #apply_op_list, loss_list = helper_buildLoss_usingOtherApproaches.build_4reluFc_Loss_forD7AndD10_updateBeforeLayers(teacher_output_dict, student_output_dict, optimizer, tvars)

        # run order: correctLabels, 3relu, fc
        #apply_op_list = []
        #loss_list = []
        #apply_op_list.append(apply_op)
        #apply_op_list, loss_list = helper_buildLoss_usingOtherApproaches.build_3relu_Loss_forD10(apply_op_list, loss_list, teacher_output_dict, student_output_dict, optimizer, tvars)
        #apply_op_list, loss_list = helper_buildLoss_usingOtherApproaches.build_fc_Loss_forD10(apply_op_list, loss_list, teacher_output_dict, student_output_dict, optimizer, tvars)

        apply_op_list, loss_list = helper_buildLoss.build_3reluFc_Loss_forD7AndD10(teacher_output_dict, student_output_dict, optimizer, tvars)
        #apply_op_list, loss_list = helper_buildLoss.build_4reluFc_Loss_forD10(teacher_output_dict, student_output_dict, optimizer, tvars)
        #apply_op_list = helper_buildLoss.build_7reluFc_Loss(teacher_output_dict, student_output_dict, optimizer, tvars)
        #apply_op_list = helper_buildLoss.build_2reluFc_Loss_forD5AndD6(teacher_output_dict, student_output_dict, optimizer, tvars)
        #apply_op_list, loss_list = helper_buildLoss.build_onlyFc_Loss(teacher_output_dict, student_output_dict, optimizer, tvars)

        #apply_op_list.append(apply_op)
        tf.logging.info("The length of apply_op_list: {}".format(len(apply_op_list)))

        train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(apply_op_list):
            train_op = tf.group(*train_ops)
        return train_op, loss_list
        #return train_op, teacherlayers, cosine_lists, maxCosine_counts

    is_training = 'train' in mode
    if is_training:
      self.global_step = tf.train.get_or_create_global_step()

    if is_training:
      logits, self.student_output_dict = build_model(images, self.num_classes, is_training, self.hparams, self.type, self.teacher_model)
    else:
      logits, self.student_output_dict = build_model(images, self.num_classes, is_training, self.hparams, self.type)

    self.predictions, self.cost = helper_utils.setup_loss(logits, labels)
    self.accuracy, self.eval_op = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(self.predictions, 1))
    self._calc_num_trainable_params()
    self.cost = helper_utils.decay_weights(self.cost, self.hparams.weight_decay_rate)

    if is_training:
      self.teacher_output_dict = self.teacher_model.output_dict

      self.train_op, self.loss_list = \
        build_loss_and_train_op(self.hparams, self.lr_rate_ph, self.cost, self.teacher_output_dict, self.student_output_dict, self.global_step)

      # adding cosineSimilarity
      #self.train_op, self.teacherlayers, self.cosine_lists, self.maxCosine_counts = build_loss_and_train_op(
      #  self.hparams, self.lr_rate_ph, self.cost, self.teacher_output_dict, self.student_output_dict, self.global_step)
      #self.teacher_group1_block3_sub1_relu = tf.nn.l2_normalize(self.teacher_output_dict["group1_block3_sub1_relu"], axis=0)
      #self.teacher_group3_block0_sub1_relu = tf.nn.l2_normalize(self.teacher_output_dict["group3_block0_sub1_relu"], axis=0)
      #self.teacher_unit_last_relu = tf.nn.l2_normalize(self.teacher_output_dict["unit_last_relu"], axis=0)

      # for visualizing the output
      #self.teacherGroup11_normFalse = self.teacher_output_dict["group1_block0_sub1_relu"]
      #self.teacherGroup21_normFalse = self.teacher_output_dict["group2_block0_sub1_relu"]
      #self.teacherGroup31_normFalse = self.teacher_output_dict["group3_block0_sub1_relu"]
      #self.teacherGroup12_normFalse = self.teacher_output_dict["group1_block0_sub2_relu"]
      #self.teacherGroup22_normFalse = self.teacher_output_dict["group2_block0_sub2_relu"]
      #self.teacherGroup32_normFalse = self.teacher_output_dict["group3_block0_sub2_relu"]
      #self.teacherLast_normFalse = self.teacher_output_dict["unit_last_relu"]

      # compared methods
      #self.train_op, self.loss_list = helper_buildLoss_comparedMethods.build_loss_and_train_op_comparedMethods \
      #  (self.hparams, self.lr_rate_ph, self.cost, self.teacher_output_dict, self.student_output_dict, self.global_step,
      #   self.predictions, self.num_classes)

      # compared methods, KD parameters from paper
      #self.train_op, self.loss_list, self.lamma_KD_ph = helper_buildLoss_comparedMethods.build_loss_and_train_op_comparedMethods \
      #  (self.hparams, self.lr_rate_ph, self.cost, self.teacher_output_dict, self.student_output_dict, self.global_step,
      #   self.predictions, self.num_classes)
      #tvars_KD = helper_buildLoss_comparedMethods.get_variables_for_KD()
      #tf.logging.info("tvars_KD: {}".format(tvars_KD))
      #self.saverKD = tf.train.Saver(tvars_KD)

      # DeCAF phase2, restore weights from DeCAF phase1
      #self.saverDeCAF = helper_buildLoss_DeCAF.define_saver_to_restore_weights_from_DeCAF_phase1()

    with tf.device('/cpu:0'):
      self.saver = tf.train.Saver(max_to_keep=2)

    tf.logging.info('trainable variables: {}'.format(len(tf.trainable_variables())))

  def _build_graph_independent(self, images, labels, mode):

    def _build_train_op(hparams, initial_lr, cost, global_step):
      # tvars = tf.trainable_variables()
      # grads = tf.gradients(cost, tvars)
      # if hparams.gradient_clipping_by_global_norm > 0.0:
      #   grads, norm = tf.clip_by_global_norm(grads, hparams.gradient_clipping_by_global_norm)
      optimizer = tf.train.MomentumOptimizer(initial_lr, 0.9, use_nesterov=True)
      grads_tvars = optimizer.compute_gradients(cost)
      return optimizer, grads_tvars

      # apply_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_step')
      # train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # with tf.control_dependencies([apply_op]):
      #   train_op = tf.group(*train_ops)
      # return train_op

    # is_training = 'train' in mode
    # if is_training:
    #   self.global_step = tf.train.get_or_create_global_step()

    logits, self.output_dict = build_model(images, self.num_classes, is_training, self.hparams, self.type)
    self.predictions, self.cost = helper_utils.setup_loss(logits, labels)
    self.accuracy, self.eval_op = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(self.predictions, 1))

    self._calc_num_trainable_params()
    self.cost = helper_utils.decay_weights(self.cost, self.hparams.weight_decay_rate)

    if is_training:
      self.optimizer, self.grads_tvars = _build_train_op(self.hparams, self.lr_rate_ph, self.cost, self.global_step)

    #with tf.device('/cpu:0'):
    #self.saver = tf.train.Saver(max_to_keep=2)
    # self.saver = tf.train.Saver()

    # for var in tf.trainable_variables():
    #   tf.logging.info(var)
    # tf.logging.info('num of trainable variables: {}'.format(len(tf.trainable_variables())))
    # # see the output of every layer, for analyze pruning
    # for key, value in sorted(self.output_dict.items()):
    #   print(key, value)
    # self.output_list = helper_output_analyze.return_output_list(self.output_dict)

class CifarModelTrainer(object):
  """Trains an instance of the CifarModel class."""

  def __init__(self, hparams):
    self._session = None
    self.hparams = hparams

    self.model_dir = os.path.join(FLAGS.checkpoint_dir, 'model')
    self.log_dir = os.path.join(FLAGS.checkpoint_dir, 'log')

    np.random.seed(0)
    if hparams.dataset == 'cifar10' or hparams.dataset == 'cifar100':
        self.data_loader = data_utils_cifar.DataSetCifar(hparams)
    if hparams.dataset == 'imagenet_256' or hparams.dataset == 'imagenet_32':
        self.data_loader = data_utils_imagenet.DataSetImageNet(hparams)
    else:
      raise ValueError("Not found dataSet name")
    np.random.seed()
    self.data_loader.reset()

  def save_model(self, step=None):
    model_save_name = os.path.join(self.model_dir, 'model.ckpt')
    if not tf.gfile.IsDirectory(self.model_dir):
      tf.gfile.MakeDirs(self.model_dir)
    self.saver.save(self.session, model_save_name, global_step=step)
    tf.logging.info('Saved sub model')

  def init_save_log_writer(self):
    if not tf.gfile.IsDirectory(self.log_dir):
      tf.gfile.MakeDirs(self.log_dir)
    self.summary_train_writer = tf.summary.FileWriter(self.log_dir+"/train", self.session.graph)
    self.summary_eval_writer = tf.summary.FileWriter(self.log_dir+"/eval")
    tf.logging.info('Init summary writers')

  def extract_model_spec(self):
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
    if checkpoint_path is not None:
      self.saver.restore(self.session, checkpoint_path)
      tf.logging.info('Loaded sub model checkpoint from %s', checkpoint_path)
    else:
      self.save_model(step=0)

  def extract_teacher_model_spec(self, model):
    teacher_model_dir = os.path.join(FLAGS.teacher_checkpoint_dir, 'model')
    teacher_checkpoint_path = tf.train.latest_checkpoint(teacher_model_dir)
    if teacher_checkpoint_path is not None:
      model.teacher_saver.restore(self.session, teacher_checkpoint_path)
      tf.logging.info('Loaded teacher model checkpoint from: {}'.format(teacher_checkpoint_path))
    else:
      raise ValueError("Not fond teacher checkpoint dir")

  def eval_teacher_model(self, model, data_loader, mode, curr_epoch):
    while True:
      try:
        with self._new_session(model):
          self.extract_teacher_model_spec(model)
          # accuracy = helper_utils.eval_child_model(self.session, model.teacher_model, self.data_loader, mode)
          self.save_model(step=curr_epoch)
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)

  def restore_and_save_teacher_model(self, model, curr_epoch):
    tf.logging.info('Begin to restore and evaluate teacher model..................................................')
    self.eval_teacher_model(model, self.data_loader, 'eval_train', curr_epoch)
    self.eval_teacher_model(model, self.data_loader, 'test', curr_epoch)
    tf.logging.info('Finish to restore and evaluate teacher model.................................................')

  def eval_child_model(self, model, data_loader, mode):
    while True:
      try:
        with self._new_session(model):
          self.init_save_log_writer()
          accuracy = helper_utils.eval_child_model(self.session, model, data_loader, mode, self.summary_eval_writer)
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)
    return accuracy

  def _compute_final_accuracies(self, meval):
    tf.logging.info('Begin to evaluate {} model..................................................'.format(meval.type))
    train_accuracy = self.eval_child_model(meval, self.data_loader, 'eval_train')
    test_accuracy = self.eval_child_model(meval, self.data_loader, 'test')
    tf.logging.info('Finish to evaluate {} model.................................................'.format(meval.type))
    return test_accuracy, train_accuracy

  def _build_models(self, hparams, images, num_classes, reuse_vars):
    """Builds the image models for train and eval."""
    if FLAGS.model_type=="dependent_student":
      tf.logging.info("build dependent student###############################")
      with tf.variable_scope('model', use_resource=False):
        m_teacher = CifarModel(self.hparams, 'teacher')
        m_teacher.build('train')
        m_student = CifarModel(self.hparams, 'dependent_student')
        m_student.build('train', m_teacher)
        self._num_trainable_params = m_student.num_trainable_params
        self._saver = m_student.saver

      with tf.variable_scope('model', reuse=True, use_resource=False):
        meval_student = CifarModel(self.hparams, 'dependent_student')
        meval_student.build('eval')
      return m_student, meval_student

    elif FLAGS.model_type == "independent_student":
      tf.logging.info("build independent student###########################")
      with tf.variable_scope('model', reuse=reuse_vars, use_resource=False):

        train_logits, output_dict = build_model(images, num_classes, True, hparams, FLAGS.model_type)
        # m = CifarModel(self.hparams, 'independent_student')
        # m.build('train', images, labels)
        # self._num_trainable_params = m.num_trainable_params
        # self._saver = m.saver
      with tf.variable_scope('model', reuse=True, use_resource=False):
        test_logits, output_dict = build_model(images, num_classes, False, hparams, FLAGS.model_type)

        # meval = CifarModel(self.hparams, 'independent_student')
        # meval.build('eval', images, labels)
      return train_logits, test_logits

    elif FLAGS.model_type == "teacher":
      tf.logging.info("build teacher###########################")
      with tf.variable_scope('model', use_resource=False):
        m = CifarModel(self.hparams, 'teacher')
        m.build('train')
        self._num_trainable_params = m.num_trainable_params
        self._saver = m.saver
      with tf.variable_scope('model', reuse=True, use_resource=False):
        meval = CifarModel(self.hparams, 'teacher')
        meval.build('eval')
      return m, meval

  def average_gradients(self, tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  def run_model(self):

    PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
    def assign_to_device(device, ps_device='/cpu:0'):
      def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
          return "/" + ps_device
        else:
          return device
      return _assign

    hparams = self.hparams
    start_time = time.time()

    with tf.device('/cpu:0'):
      tower_grads = []
      reuse_vars = False

      if FLAGS.dataset == 'cifar10':
        image_size=32
        num_classes=10
      elif FLAGS.dataset == 'imagenet_256':
        image_size = 256
        num_classes = 700
      elif FLAGS.dataset == 'imagenet_32':
        image_size = 32
        num_classes = 11
      else:
        raise EOFError("NOt found dataset!")
      images = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, 3], name="images_placeholder")
      labels = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="labels_placeholder")

      # Build the graph
      for i in range(FLAGS.num_gpus):
        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):

            sub_images = images[i * hparams.batch_size: (i+1) * hparams.batch_size]
            sub_labels = labels[i * hparams.batch_size: (i+1) * hparams.batch_size]

            train_logits, test_logits = self._build_models(hparams, sub_images, num_classes, reuse_vars)
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=sub_labels))
            loss_op = helper_utils.decay_weights(loss_op, hparams.weight_decay_rate)
            lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr_rate_ph, momentum=0.9, use_nesterov=True)
            grads_tvars = optimizer.compute_gradients(loss_op)
            if i == 0:
              correct_pred = tf.equal(tf.argmax(test_logits, 1), tf.argmax(sub_labels, 1))
              accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            reuse_vars = True
            tower_grads.append(grads_tvars)

      tower_grads = self.average_gradients(tower_grads)
      apply_op = optimizer.apply_gradients(tower_grads)
      train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies([apply_op]):
        train_op = tf.group(*train_ops)

      training_accuracy_list = []
      test_accuracy_list = []
      steps_per_epoch = int(hparams.train_size / (hparams.batch_size * FLAGS.num_gpus))
      total_steps = hparams.num_epochs * steps_per_epoch
      tf.logging.info('Steps per epoch: {}'.format(steps_per_epoch))
      tf.logging.info("Total_steps {}".format(total_steps))

      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      with tf.Session() as session:
        session.run(init)
        for curr_step in range(1, total_steps + 1):
          start_epoch_time = time.time()

          curr_lr = helper_utils.get_lr(hparams=hparams, t_cur=curr_step)
          lr_rate_ph.load(curr_lr, session=session)

          train_images, train_labels = self.data_loader.next_batch(FLAGS.num_gpus)
          ts = time.time()
          session.run(train_op,feed_dict={images: train_images,labels: train_labels})
          te = time.time() - ts

          if curr_step % 100 == 0 or curr_step == 1:
            loss, acc = session.run([loss_op, accuracy], feed_dict={images: train_images, labels: train_labels})
            print("Step " + str(curr_step) + ": Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc) + ", %i Examples/sec" % int(len(train_images) / te) + ", lr: "+ str(curr_lr))

          if curr_step % steps_per_epoch == 0 or curr_step == total_steps-1:
            curr_epoch = int(curr_step / steps_per_epoch)
            tf.logging.info("curr_step: {}, curr_epoch: {}, lr: {}".format(curr_step, curr_epoch, curr_lr))

            test_accuracy_per_epoch = np.mean([session.run(accuracy, feed_dict={
              images: self.data_loader.test_images[i:i + hparams.batch_size],
              labels: self.data_loader.test_labels[i:i + hparams.batch_size]})
                                     for i in range(0, len(self.data_loader.test_labels), hparams.batch_size)])
            test_accuracy_list.append(test_accuracy_per_epoch)
            training_accuracy_list.append(acc)
            print("Testing Accuracy: {}".format(test_accuracy_per_epoch))
            print("Training Accuracy List: {}".format(training_accuracy_list))
            print("Testing Accuracy List: {}".format(test_accuracy_list))
            print('Epoch time(min): {}\n'.format((time.time() - start_epoch_time) / 60.0))
            start_epoch_time = time.time()

    end_time = time.time()
    runtime = round((end_time - start_time) / (60 * 60), 2)
    tf.logging.info("run time is: " + str(runtime) + " hour")

  @property
  def saver(self):
    return self._saver

  # @property
  # def session(self):
  #   return self._session

  @property
  def num_trainable_params(self):
    return self._num_trainable_params


def main(_):

  if FLAGS.dataset not in ['cifar10', 'imagenet_256', 'imagenet_32']:
    raise ValueError('Invalid dataset: %s' % FLAGS.dataset)

  hparams = tf.contrib.training.HParams(
      train_size=FLAGS.train_size,
      test_size=FLAGS.test_size,
      validation_size=0,
      eval_test=1,
      dataset=FLAGS.dataset,
      data_path=FLAGS.data_path,
      batch_size=FLAGS.batch_size,
      gradient_clipping_by_global_norm=5.0)

  if 'wrn' in FLAGS.model_name:
    hparams.add_hparam('model_name', str(FLAGS.model_name))
    hparams.add_hparam('num_epochs', 200)
    hparams.add_hparam('wrn_size', 160)
    hparams.add_hparam('lr', 0.1)
    hparams.add_hparam('weight_decay_rate', 5e-4)
    hparams.add_hparam('lamma_KD_initial', FLAGS.lamma_KD_initial)
    hparams.add_hparam('KD_checkpoint_dir', str(FLAGS.KD_checkpoint_dir))
    hparams.add_hparam('DeCAF_checkpoint_dir', str(FLAGS.DeCAF_checkpoint_dir))
    hparams.add_hparam('test_data_type', str(FLAGS.test_data_type))
    hparams.add_hparam('teacher_model_name', str(FLAGS.teacher_model_name))
    hparams.add_hparam('checkpoint_dir', str(FLAGS.checkpoint_dir))
    # hparams.add_hparam('lr_decay_epoch', '[60, 120, 160]')
  else:
    raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)

  cifar_trainer = CifarModelTrainer(hparams)
  cifar_trainer.run_model()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
