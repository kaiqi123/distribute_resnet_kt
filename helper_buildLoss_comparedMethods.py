import tensorflow as tf

def embed_studentLayer_byConv(layerName, inputs, out_filter):
    with tf.variable_scope('embed_'+layerName):
        in_filter = inputs.get_shape()[3:].as_list()[0]
        weights = tf.Variable(tf.random_normal([3, 3, in_filter, out_filter], stddev=1e-2), trainable=True,name='student_embed_weights')
        biases = tf.Variable(tf.zeros(out_filter), trainable=True, name='student_embed_biases')
        outputs = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
        outputs = tf.nn.bias_add(outputs, biases)
        return outputs

def build_loss_noNorm(teacher_layer, student_layer, tvars_layer, optimizer):
    loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_layer, student_layer))))
    grads_layer = tf.gradients(loss_layer, tvars_layer)
    apply_op_layer = optimizer.apply_gradients(zip(grads_layer, tvars_layer))
    return loss_layer, apply_op_layer

# HT: phase 1 of intermediate representations KT technique (single layer)
def build_HT_loss(teacher_output_dict, student_output_dict, optimizer, tvars):
    tf.logging.info("build_HT_loss")
    tvars_init = [var for var in tvars if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    tvars_group1 = [var for var in tvars if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    tvars_HT =  tvars_init + tvars_group1

    # conv on group2_block0_sub1_relu of student to make the same width with teacher
    layerName = "group2_block0_sub1_relu"
    teacher_layer = teacher_output_dict[layerName]
    student_layer = student_output_dict[layerName]
    if student_layer.get_shape()[3] != teacher_layer.get_shape()[3]:
        tf.logging.info("Conv on student: {}".format(layerName))
        student_layer = embed_studentLayer_byConv(layerName, student_layer, teacher_layer.get_shape()[3:].as_list()[0])
        tvars_embed = [var for var in tf.trainable_variables() if var.op.name.startswith("model/embed_group2_block0_sub1_relu")]
        tvars_HT = tvars_HT + tvars_embed

    loss_HT, apply_op_HT = build_loss_noNorm(teacher_layer, student_layer, tvars_HT, optimizer)
    tf.logging.info("tvars_HT: {}".format(tvars_HT))

    apply_op_list = [apply_op_HT]
    loss_list = [loss_HT]
    return apply_op_list, loss_list

# hard_logits KT technique: where hard_logits of teacher are transferred to student softmax output
def build_hard_logits(teacher_output_dict, student_output_dict, optimizer, tvars, predictions, num_classes):
    tf.logging.info("build_hard_logits")
    ind_max = tf.argmax(teacher_output_dict["fc"], axis=1)
    hard_logits = tf.one_hot(ind_max, num_classes)
    loss_hardLogits = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(hard_logits, predictions))))
    grads_hardLogits = tf.gradients(loss_hardLogits, tvars)
    apply_op_hardLogits = optimizer.apply_gradients(zip(grads_hardLogits, tvars))
    apply_op_list = [apply_op_hardLogits]
    loss_list = [loss_hardLogits]
    return apply_op_list, loss_list

# fromFitNetPaper: Soft Logits KT technique, phase 2 of intermediate representations KT technique (single layer)
def build_KD_loss_fromFitNetPaper(teacher_output_dict, student_output_dict, optimizer, tvars, cost):
    tf.logging.info("build_KD_loss_fromFitNetPaper")
    lamma_KD = tf.Variable(4.0, name='lamma_KD', trainable=False)
    t = 3.0
    teacher_softmax = tf.nn.softmax(teacher_output_dict["fc"]/t)
    student_softmax = tf.nn.softmax(student_output_dict["fc"]/t)
    loss_softmax = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_softmax, student_softmax))))
    loss_KD = cost + lamma_KD*loss_softmax
    grads_KD = tf.gradients(loss_KD, tvars)
    apply_op_KD = optimizer.apply_gradients(zip(grads_KD, tvars))
    apply_op_list = [apply_op_KD]
    loss_list = [loss_KD]
    return apply_op_list, loss_list, lamma_KD

# Soft Logits KT technique, phase 2 of intermediate representations KT technique (single layer)
def build_KD_loss(teacher_output_dict, student_output_dict, optimizer, tvars, student_softmax, cost):
    tf.logging.info("build_KD_loss")
    teacher_softmax = tf.nn.softmax(teacher_output_dict["fc"])
    loss_softmax = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_softmax, student_softmax))))
    loss_KD = 0.2*cost + loss_softmax
    grads_KD = tf.gradients(loss_KD, tvars)
    apply_op_KD = optimizer.apply_gradients(zip(grads_KD, tvars))
    apply_op_list = [apply_op_KD]
    loss_list = [loss_KD]
    return apply_op_list, loss_list

def build_loss_and_train_op_comparedMethods(hparams, initial_lr, cost, teacher_output_dict, student_output_dict, global_step, predictions, num_classes):
    for key, v in sorted(teacher_output_dict.items()):
      tf.logging.info('teacher_output_dict, key: {}, shape: {}'.format(key, v.shape))
    for key, v in sorted(student_output_dict.items()):
      tf.logging.info('student_output_dict, key: {}, shape: {}'.format(key, v.shape))

    optimizer = tf.train.MomentumOptimizer(initial_lr, 0.9, use_nesterov=True)
    tvars = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/")]
    for tvar in tvars: tf.logging.info('tvar: {}'.format(tvar))
    tf.logging.info('model/student_architecture/  trainable variables: {}'.format(len(tvars)))

    # HT: phase 1 of intermediate representations KT technique (single layer), train 100 epochs
    #apply_op_list, loss_list = build_HT_loss(teacher_output_dict, student_output_dict, optimizer, tvars)

    # Soft Logits KT technique, phase 2 of intermediate representations KT technique (single layer)
    #apply_op_list, loss_list = build_KD_loss(teacher_output_dict, student_output_dict, optimizer, tvars, predictions, cost)
    apply_op_list, loss_list, lamma_KD = build_KD_loss_fromFitNetPaper(teacher_output_dict, student_output_dict, optimizer, tvars, cost)

    # hard_logits KT technique: where hard_logits of teacher are transferred to student softmax output
    #apply_op_list, loss_list = build_hard_logits(teacher_output_dict, student_output_dict, optimizer, tvars, predictions, num_classes)

    tf.logging.info("The length of apply_op_list: {}".format(len(apply_op_list)))

    train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(apply_op_list):
        train_op = tf.group(*train_ops)
    #return train_op, loss_list
    return train_op, loss_list, lamma_KD

def get_variables_for_KD():
    tvars_init = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/init") and var.op.name.endswith("weights")]
    tvars_group1 = [var for var in tf.trainable_variables() if var.op.name.startswith("model/student_architecture/group_1") and var.op.name.endswith("weights")]
    tvars_HT =  tvars_init + tvars_group1
    return tvars_HT