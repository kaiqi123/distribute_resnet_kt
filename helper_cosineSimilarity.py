import tensorflow as tf
import custom_ops as ops

def cosineSimilarity(data1, data2):
    with tf.variable_scope("cosine"):
        product12 = tf.reduce_sum(tf.multiply(data1, data2))
        data1_sqrt = tf.sqrt(tf.reduce_sum(tf.square(data1)))
        data2_sqrt = tf.sqrt(tf.reduce_sum(tf.square(data2)))
        cosine = tf.divide(product12, tf.multiply(data1_sqrt, data2_sqrt))
    return cosine

def cosineSimilarity_of_OneGroup(studentGroupName, teacherGroup, teacher_output_dict, student_output_dict):
    student_group2_block0_sub1_relu = tf.nn.l2_normalize(student_output_dict[str(studentGroupName)], axis=0)
    teacher_block0_sub2_relu = tf.nn.l2_normalize(teacher_output_dict[str(teacherGroup)+"_block0_sub2_relu"], axis=0)
    teacher_block1_sub1_relu = tf.nn.l2_normalize(teacher_output_dict[str(teacherGroup)+"_block1_sub1_relu"], axis=0)
    teacher_block1_sub2_relu = tf.nn.l2_normalize(teacher_output_dict[str(teacherGroup)+"_block1_sub2_relu"], axis=0)
    teacher_block2_sub1_relu = tf.nn.l2_normalize(teacher_output_dict[str(teacherGroup)+"_block2_sub1_relu"], axis=0)
    teacher_block2_sub2_relu = tf.nn.l2_normalize(teacher_output_dict[str(teacherGroup)+"_block2_sub2_relu"], axis=0)
    teacher_block3_sub1_relu = tf.nn.l2_normalize(teacher_output_dict[str(teacherGroup)+"_block3_sub1_relu"], axis=0)
    teacher_block3_sub2_relu = tf.nn.l2_normalize(teacher_output_dict[str(teacherGroup)+"_block3_sub2_relu"], axis=0)
    teacher_group2_block0_sub1_relu = tf.nn.l2_normalize(teacher_output_dict[str(studentGroupName)], axis=0)
    teacher_reluOutput_list = [teacher_block0_sub2_relu, teacher_block1_sub1_relu,
                               teacher_block1_sub2_relu, teacher_block2_sub1_relu,
                               teacher_block2_sub2_relu, teacher_block3_sub1_relu,
                               teacher_block3_sub2_relu, teacher_group2_block0_sub1_relu]
    #teacher_reluName_list = [str(teacherGroup)+"_block0_sub2_relu", str(teacherGroup)+"_block1_sub1_relu",
    #                         str(teacherGroup)+"_block1_sub2_relu", str(teacherGroup)+"_block2_sub1_relu",
    #                         str(teacherGroup)+"_block2_sub2_relu", str(teacherGroup)+"_block3_sub1_relu",
    #                         str(teacherGroup)+"_block3_sub2_relu", str(studentGroupName)]

    if student_group2_block0_sub1_relu.get_shape()[3] != teacher_block0_sub2_relu.get_shape()[3]:
        tf.logging.info("Zero padding on student: {}" .format(studentGroupName))
        student_group2_block0_sub1_relu = ops.zero_pad(student_group2_block0_sub1_relu, student_group2_block0_sub1_relu.get_shape()[3], teacher_block0_sub2_relu.get_shape()[3])

    cosine_list = []
    cosine_max = tf.constant(0.0)
    maxCosine_teacherlayer = teacher_block0_sub2_relu
    maxCosine_count = 0
    assert len(teacher_reluOutput_list) == 8
    for i in range(len(teacher_reluOutput_list)):
        cosine = cosineSimilarity(student_group2_block0_sub1_relu, teacher_reluOutput_list[i])
        cosine_list.append(cosine)
        def return_cosine(teacherlayer, maxCosine_count):
            return cosine, teacherlayer, maxCosine_count
        def return_cosine_max(teacherlayer, maxCosine_count):
            return cosine_max, teacherlayer, maxCosine_count
        cosine_max, maxCosine_teacherlayer, maxCosine_count = tf.cond(cosine > cosine_max,
            lambda: return_cosine(teacher_reluOutput_list[i], i),
            lambda: return_cosine_max(maxCosine_teacherlayer, maxCosine_count))

    #cosine_max_test = tf.reduce_max(cosine_list, reduction_indices=[0])
    #cosine_index = tf.argmax(cosine_list, axis=0)
    #print(len(cosine_list), cosine_max_test, cosine_index)
    return student_group2_block0_sub1_relu, maxCosine_teacherlayer, cosine_list, tf.convert_to_tensor(maxCosine_count)

def build_loss_afterCosine(norm_student, norm_teacher, tvars_layer, optimizer):
    loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(norm_teacher, norm_student))))
    grads_layer = tf.gradients(loss_layer, tvars_layer)
    apply_op_layer = optimizer.apply_gradients(zip(grads_layer, tvars_layer))
    return loss_layer, apply_op_layer

def build_loss_normal(layerName, tvars_layer, teacher_output_dict, student_output_dict, optimizer):
    norm_teacher = tf.nn.l2_normalize(teacher_output_dict[layerName], axis=0)
    norm_student = tf.nn.l2_normalize(student_output_dict[layerName], axis=0)
    if norm_student.get_shape()[3] != norm_teacher.get_shape()[3]:
        tf.logging.info("Zero padding on student: {}".format(layerName))
        norm_student = ops.zero_pad(norm_student, norm_student.get_shape()[3], norm_teacher.get_shape()[3])
    loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(norm_teacher, norm_student))))
    grads_layer = tf.gradients(loss_layer, tvars_layer)
    apply_op_layer = optimizer.apply_gradients(zip(grads_layer, tvars_layer))
    return loss_layer, apply_op_layer

def build_4reluFc_Loss_forD10_usingCosine(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("build_4reluFc_Loss_forD10_usingCosine")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    tvars_init = [var for var in tvars if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    loss_init, apply_op_init = build_loss_normal("group1_block0_sub1_relu", tvars_init, teacher_output_dict, student_output_dict, optimizer)
    tf.logging.info("tvars_init: {}".format(tvars_init))

    student_group2_block0_sub1_relu, teacherlayer1, cosine_list1, maxCosine_count1 = cosineSimilarity_of_OneGroup("group2_block0_sub1_relu", "group1", teacher_output_dict, student_output_dict)
    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    loss_group1, apply_op_group1 = build_loss_afterCosine(student_group2_block0_sub1_relu, teacherlayer1, tvars_group1, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    student_group3_block0_sub1_relu, teacherlayer2, cosine_list2, maxCosine_count2 = cosineSimilarity_of_OneGroup("group3_block0_sub1_relu", "group2", teacher_output_dict, student_output_dict)
    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    loss_group2, apply_op_group2 = build_loss_afterCosine(student_group3_block0_sub1_relu, teacherlayer2, tvars_group2, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))

    student_unit_last_relu, teacherlayer3, cosine_list3, maxCosine_count3 = cosineSimilarity_of_OneGroup("unit_last_relu", "group3", teacher_output_dict, student_output_dict)
    tvars_group3 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("weights")]
    loss_last_relu, apply_op_group3 = build_loss_afterCosine(student_unit_last_relu, teacherlayer3, tvars_group3, optimizer)
    tf.logging.info("tvars_group3: {}".format(tvars_group3))

    apply_op_list = [apply_op_init, apply_op_group1, apply_op_group2, apply_op_group3, apply_op_fc]

    teacherlayers = [teacherlayer1, teacherlayer2, teacherlayer3]
    cosine_lists = [cosine_list1, cosine_list2, cosine_list3]
    maxCosine_counts = [maxCosine_count1, maxCosine_count2, maxCosine_count3]

    return apply_op_list, teacherlayers, cosine_lists, maxCosine_counts


def build_3reluFc_Loss_forD10_usingCosine(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("build_3reluFc_Loss_forD10_usingCosine")
    loss_fc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_output_dict["fc"], student_output_dict["fc"]))))
    grads_fc = tf.gradients(loss_fc, tvars)
    apply_op_fc = optimizer.apply_gradients(zip(grads_fc, tvars))

    student_group2_block0_sub1_relu, teacherlayer1, cosine_list1, maxCosine_count1 = cosineSimilarity_of_OneGroup("group2_block0_sub1_relu", "group1", teacher_output_dict, student_output_dict)
    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    loss_group1, apply_op_group1 = build_loss_afterCosine(student_group2_block0_sub1_relu, teacherlayer1, tvars_group1, optimizer)
    tf.logging.info("tvars_group1: {}".format(tvars_group1))

    student_group3_block0_sub1_relu, teacherlayer2, cosine_list2, maxCosine_count2 = cosineSimilarity_of_OneGroup("group3_block0_sub1_relu", "group2", teacher_output_dict, student_output_dict)
    tvars_group2 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_2") and var.op.name.endswith("weights")]
    loss_group2, apply_op_group2 = build_loss_afterCosine(student_group3_block0_sub1_relu, teacherlayer2, tvars_group2, optimizer)
    tf.logging.info("tvars_group2: {}".format(tvars_group2))

    student_unit_last_relu, teacherlayer3, cosine_list3, maxCosine_count3 = cosineSimilarity_of_OneGroup("unit_last_relu", "group3", teacher_output_dict, student_output_dict)
    tvars_group3 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_3") and var.op.name.endswith("weights")]
    loss_last_relu, apply_op_group3 = build_loss_afterCosine(student_unit_last_relu, teacherlayer3, tvars_group3, optimizer)
    tf.logging.info("tvars_group3: {}".format(tvars_group3))

    apply_op_list = [apply_op_group1, apply_op_group2, apply_op_group3, apply_op_fc]

    teacherlayers = [teacherlayer1, teacherlayer2, teacherlayer3]
    cosine_lists = [cosine_list1, cosine_list2, cosine_list3]
    maxCosine_counts = [maxCosine_count1, maxCosine_count2, maxCosine_count3]

    return apply_op_list, teacherlayers, cosine_lists, maxCosine_counts





