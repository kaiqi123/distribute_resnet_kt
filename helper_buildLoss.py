import tensorflow as tf
import custom_ops as ops

def build_loss(layerName, tvars_layer, teacher_output_dict, student_output_dict, optimizer):
    norm_teacher = tf.nn.l2_normalize(teacher_output_dict[layerName], axis=0)
    norm_student = tf.nn.l2_normalize(student_output_dict[layerName], axis=0)
    if norm_student.get_shape()[3] != norm_teacher.get_shape()[3]:
        tf.logging.info("Zero padding on student: {}".format(layerName))
        norm_student = ops.zero_pad(norm_student, norm_student.get_shape()[3], norm_teacher.get_shape()[3])
    loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(norm_teacher, norm_student))))
    grads_layer = tf.gradients(loss_layer, tvars_layer)
    apply_op_layer = optimizer.apply_gradients(zip(grads_layer, tvars_layer))
    return loss_layer, apply_op_layer


def build_7reluFc_Loss(teacher_output_dict, student_output_dict, optimizer, tvars):
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    tvars_conv0 = [var for var in tvars if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    loss_group1_relu1, apply_op_conv0 = build_loss("group1_block0_sub1_relu", tvars_conv0, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_conv0: {}".format(tvars_conv0))

    tvars_group1_conv1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("conv1/weights")]
    loss_group1_relu2, apply_op_group1_conv1 = build_loss("group1_block0_sub2_relu", tvars_group1_conv1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1_conv1: {}".format(tvars_group1_conv1))

    tvars_group1_conv2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("conv2/weights")]
    loss_group2_relu1, apply_op_group1_conv2 = build_loss("group2_block0_sub1_relu", tvars_group1_conv2, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1_conv2: {}".format(tvars_group1_conv2))

    tvars_group2_conv1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("conv1/weights")]
    loss_group2_relu2, apply_op_group2_conv1 = build_loss("group2_block0_sub2_relu", tvars_group2_conv1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2_conv1: {}".format(tvars_group2_conv1))

    tvars_group2_conv2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("conv2/weights")]
    loss_group3_relu1, apply_op_group2_conv2 = build_loss("group3_block0_sub1_relu", tvars_group2_conv1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2_conv2: {}".format(tvars_group2_conv2))

    tvars_group3_conv1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("conv1/weights")]
    loss_group3_relu2, apply_op_group3_conv1 = build_loss("group3_block0_sub2_relu", tvars_group2_conv1,teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group3_conv1: {}".format(tvars_group3_conv1))

    tvars_group3_conv2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("conv2/weights")]
    loss_last_relu, apply_op_group3_conv2 = build_loss("unit_last_relu", tvars_group2_conv1, teacher_output_dict,student_output_dict, optimizer)
    tf.logging.info("tvars_group3_conv2: {}".format(tvars_group3_conv2))

    apply_op_list = [apply_op_conv0, apply_op_group1_conv1, apply_op_group1_conv2, apply_op_group2_conv1,
                     apply_op_group2_conv2, apply_op_group3_conv1, apply_op_group3_conv2, apply_op_fc]
    return apply_op_list


def build_3reluFc_Loss_forD7AndD10(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_3reluFc_Loss_forD7AndD10 .............................................")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    loss_group1, apply_op_group1 = build_loss("group2_block0_sub1_relu", tvars_group1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    loss_group2, apply_op_group2 = build_loss("group3_block0_sub1_relu", tvars_group2, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))

    tvars_group3 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("weights")]
    loss_last_relu, apply_op_group3 = build_loss("unit_last_relu", tvars_group3, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group3: {}".format(tvars_group3))

    apply_op_list = [apply_op_group1, apply_op_group2, apply_op_group3, apply_op_fc]
    loss_list = [loss_group1, loss_group2, loss_last_relu, loss_fc]
    return apply_op_list, loss_list


def build_2reluFc_Loss_forD5AndD6(teacher_output_dict, student_output_dict, optimizer, tvars):
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    tvars_init = [var for var in tvars if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    loss_group1, apply_op_init = build_loss("group1_block0_sub1_relu", tvars_init, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_init: {}".format(tvars_init))

    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    loss_last_relu, apply_op_group1 = build_loss("unit_last_relu", tvars_group1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    apply_op_list = [apply_op_init, apply_op_group1, apply_op_fc]
    return apply_op_list

def build_4reluFc_Loss_forD10(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_4reluFc_Loss_forD10 .............................................")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    tvars_init = [var for var in tvars if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    loss_init, apply_op_init = build_loss("group1_block0_sub1_relu", tvars_init, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_init: {}".format(tvars_init))

    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    loss_group1, apply_op_group1 = build_loss("group2_block0_sub1_relu", tvars_group1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    loss_group2, apply_op_group2 = build_loss("group3_block0_sub1_relu", tvars_group2, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))

    tvars_group3 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("weights")]
    loss_last_relu, apply_op_group3 = build_loss("unit_last_relu", tvars_group3, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group3: {}".format(tvars_group3))

    apply_op_list = [apply_op_init, apply_op_group1, apply_op_group2, apply_op_group3, apply_op_fc]
    loss_list = [loss_init, loss_group1, loss_group2, loss_last_relu, loss_fc]
    return apply_op_list, loss_list

def build_onlyFc_Loss(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_onlyFc_Loss.............................................")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    apply_op_list = [apply_op_fc]
    loss_list = [loss_fc]
    return apply_op_list, loss_list

