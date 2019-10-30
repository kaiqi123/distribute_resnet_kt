import tensorflow as tf
import custom_ops as ops

def build_loss_usingAvgFc(layerName, tvars_layer, teacher_output_dict, student_output_dict, optimizer):
    #teacher_avg = ops.global_avg_pool(teacher_output_dict[layerName])
    teacher_logits = ops.fc(teacher_output_dict[layerName], 10)

    #student_avg = ops.global_avg_pool(student_output_dict[layerName])
    student_logits = ops.fc(student_output_dict[layerName], 10)

    loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_logits, student_logits))))
    grads_layer = tf.gradients(loss_layer, tvars_layer)
    apply_op_layer = optimizer.apply_gradients(zip(grads_layer, tvars_layer))
    return loss_layer, apply_op_layer

def build_4fc_Loss_forD10(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_4fc_Loss_forD10 using fc.............................................")

    tvars_init = [var for var in tvars if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    loss_init, apply_op_init = build_loss_usingAvgFc("group1_block0_sub1_relu", tvars_init, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_init: {}".format(tvars_init))

    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    tvars_group1 = tvars_init + tvars_group1
    loss_group1, apply_op_group1 = build_loss_usingAvgFc("group2_block0_sub1_relu", tvars_group1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    tvars_group2 = tvars_group1 + tvars_group2
    loss_group2, apply_op_group2 = build_loss_usingAvgFc("group3_block0_sub1_relu", tvars_group2, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))

    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    apply_op_list = [apply_op_init, apply_op_group1, apply_op_group2, apply_op_fc]
    loss_list = [loss_init, loss_group1, loss_group2, loss_fc]
    return apply_op_list, loss_list

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

def build_3OutputOfGroupFc_Loss_forD10(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_3OutputOfGroupFc_Loss_forD10 .............................................")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    loss_group1, apply_op_group1 = build_loss("group1_out", tvars_group1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    loss_group2, apply_op_group2 = build_loss("group2_out", tvars_group2, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))

    tvars_group3 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("weights")]
    loss_last_relu, apply_op_group3 = build_loss("group3_out", tvars_group3, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group3: {}".format(tvars_group3))

    apply_op_list = [apply_op_group1, apply_op_group2, apply_op_group3, apply_op_fc]
    loss_list = [loss_group1, loss_group2, loss_last_relu, loss_fc]
    return apply_op_list, loss_list

def build_4reluFc_Loss_forD7AndD10_updateBeforeLayers(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_4reluFc_Loss_forD7AndD10_updateBeforeLayers .............................................")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    tvars_init = [var for var in tvars if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    loss_init, apply_op_init = build_loss("group1_block0_sub1_relu", tvars_init, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_init: {}".format(tvars_init))

    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    tvars_group1 = tvars_init + tvars_group1
    loss_group1, apply_op_group1 = build_loss("group2_block0_sub1_relu", tvars_group1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    tvars_group2 = tvars_group1 + tvars_group2
    loss_group2, apply_op_group2 = build_loss("group3_block0_sub1_relu", tvars_group2, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))

    tvars_group3 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("weights")]
    tvars_group3 = tvars_group2 + tvars_group3
    loss_last_relu, apply_op_group3 = build_loss("unit_last_relu", tvars_group3, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group3: {}".format(tvars_group3))

    apply_op_list = [apply_op_init, apply_op_group1, apply_op_group2, apply_op_group3, apply_op_fc]
    loss_list = [loss_init, loss_group1, loss_group2, loss_last_relu, loss_fc]
    return apply_op_list, loss_list

def build_Fc3relu_Loss_forD7AndD10(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_Fc3relu_Loss_forD7AndD10 .............................................")
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

    apply_op_list = [apply_op_fc, apply_op_group1, apply_op_group2, apply_op_group3]
    loss_list = [loss_group1, loss_group2, loss_last_relu, loss_fc]
    return apply_op_list, loss_list


def build_3relu_Loss_forD10(apply_op_list, loss_list, teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_3relu_Loss_forD10 .............................................")

    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    loss_group1, apply_op_group1 = build_loss("group2_block0_sub1_relu", tvars_group1, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))
    apply_op_list.append(apply_op_group1)
    loss_list.append(loss_group1)

    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    loss_group2, apply_op_group2 = build_loss("group3_block0_sub1_relu", tvars_group2, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))
    apply_op_list.append(apply_op_group2)
    loss_list.append(loss_group2)

    tvars_group3 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("weights")]
    loss_last_relu, apply_op_group3 = build_loss("unit_last_relu", tvars_group3, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_group3: {}".format(tvars_group3))
    apply_op_list.append(apply_op_group3)
    loss_list.append(loss_last_relu)
    return apply_op_list, loss_list


def build_fc_Loss_forD10(apply_op_list, loss_list, teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("Begin to build_fc_Loss_forD10 .............................................")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))
    apply_op_list.append(apply_op_fc)
    loss_list.append(loss_fc)
    return apply_op_list, loss_list