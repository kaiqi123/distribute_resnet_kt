import tensorflow as tf
import custom_ops as ops


def build_train_op_DeCAF_onlyUpdateFc(hparams, initial_lr, cost, global_step):
    """
    DeCAF: only update Fc weights and biases using correct labels, no weight decay
    """
    print("build_train_op_DeCAF_onlyUpdateFc")
    variables_FC = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/unit_last/FC/")]
    tf.logging.info("variables_FC: {}".format(variables_FC))

    grads_fc = tf.gradients(cost, variables_FC)
    optimizer = tf.train.MomentumOptimizer(initial_lr, 0.9, use_nesterov=True)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, variables_FC))

    train_op = [apply_op_fc]
    return train_op


def build_train_op_DeCAF_UpdateFcAndBn(hparams, initial_lr, cost, global_step):
    """
    DeCAF: update Fc weights and biases, and moving_mean and moving_variance of bn, using correct labels, no weight decay
    """
    print("build_train_op_DeCAF_UpdateFcAndBn")
    variables_FC = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/unit_last/FC/")]
    tf.logging.info("variables_FC: {}".format(variables_FC))

    grads_fc = tf.gradients(cost, variables_FC)
    optimizer = tf.train.MomentumOptimizer(initial_lr, 0.9, use_nesterov=True)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, variables_FC))

    train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([apply_op_fc]):
        train_op = tf.group(*train_ops)
    return train_op


def build_train_op_DeCAF_onlyUpdateFc_usingTeacherFc(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_train_op_DeCAF_onlyUpdateFc_usingTeacherFc.............................................")
    variables_FC = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/unit_last/FC/")]
    tf.logging.info("variables_FC: {}".format(variables_FC))

    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, variables_FC)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, variables_FC))

    apply_op_list = [apply_op_fc]
    loss_list = [loss_fc]
    return apply_op_list, loss_list


def define_saver_to_restore_weights_from_DeCAF_phase1():
    tvars_student = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/")]
    for var in tvars_student:
        print(var)
    tf.logging.info("num of DeCAF_variables_to_restore: {}".format(len(tvars_student)))
    saverDeCAF = tf.train.Saver(tvars_student)
    return saverDeCAF
