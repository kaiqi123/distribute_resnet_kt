import tensorflow as tf

def initialize_group123_gammaBeta(session):
    tf.logging.info("Begin to initialize gamma and beta.................................")
    tvars = tf.trainable_variables()
    tvars_teacher_group1block0gamma1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_1/unit_1_0/") and var.op.name.endswith("init_bn/gamma")][0]
    tvars_teacher_group1block0beta1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_1/unit_1_0/") and var.op.name.endswith("init_bn/beta")][0]
    tvars_teacher_group1block0gamma2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_1/unit_1_0/") and var.op.name.endswith("bn2/gamma")][0]
    tvars_teacher_group1block0beta2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_1/unit_1_0/") and var.op.name.endswith("bn2/beta")][0]
    tvars_teacher_group2block0gamma1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_2/unit_2_0/") and var.op.name.endswith("init_bn/gamma")][0]
    tvars_teacher_group2block0beta1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_2/unit_2_0/") and var.op.name.endswith("init_bn/beta")][0]
    tvars_teacher_group2block0gamma2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_2/unit_2_0/") and var.op.name.endswith("bn2/gamma")][0]
    tvars_teacher_group2block0beta2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_2/unit_2_0/") and var.op.name.endswith("bn2/beta")][0]
    tvars_teacher_group3block0gamma1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_3/unit_3_0/") and var.op.name.endswith("init_bn/gamma")][0]
    tvars_teacher_group3block0beta1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_3/unit_3_0/") and var.op.name.endswith("init_bn/beta")][0]
    tvars_teacher_group3block0gamma2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_3/unit_3_0/") and var.op.name.endswith("bn2/gamma")][0]
    tvars_teacher_group3block0beta2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_3/unit_3_0/") and var.op.name.endswith("bn2/beta")][0]

    for var in tf.global_variables():
        if var.op.name.startswith("model/student_architecture/group_1/unit_1_0/shared_activation/init_bn/gamma"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group1block0gamma1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_1/unit_1_0/shared_activation/init_bn/beta"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group1block0beta1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_1/unit_1_0/sub2/bn2/gamma"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group1block0gamma2.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_1/unit_1_0/sub2/bn2/beta"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group1block0beta2.eval(session=session)).eval(session=session)

        if var.op.name.startswith("model/student_architecture/group_2/unit_2_0/residual_only_activation/init_bn/gamma"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group2block0gamma1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_2/unit_2_0/residual_only_activation/init_bn/beta"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group2block0beta1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_2/unit_2_0/sub2/bn2/gamma"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group2block0gamma2.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_2/unit_2_0/sub2/bn2/beta"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group2block0beta2.eval(session=session)).eval(session=session)

        if var.op.name.startswith("model/student_architecture/group_3/unit_3_0/residual_only_activation/init_bn/gamma"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group3block0gamma1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_3/unit_3_0/residual_only_activation/init_bn/beta"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group3block0beta1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_3/unit_3_0/sub2/bn2/gamma"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group3block0gamma2.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_3/unit_3_0/sub2/bn2/beta"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group3block0beta2.eval(session=session)).eval(session=session)


def initialize_conv_weights(session):
    tf.logging.info("Begin to initialize all conv weights.................................")
    tvars = tf.trainable_variables()
    tvars_teacher_conv1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/init/init_conv/weights")][0]
    tvars_teacher_group1block0conv1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_1/unit_1_0/sub1/conv1/weights")][0]
    tvars_teacher_group1block0conv2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_1/unit_1_0/sub2/conv2/weights")][0]
    tvars_teacher_group2block0conv1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_2/unit_2_0/sub1/conv1/weights")][0]
    tvars_teacher_group2block0conv2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_2/unit_2_0/sub2/conv2/weights")][0]
    tvars_teacher_group3block0conv1 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_3/unit_3_0/sub1/conv1/weights")][0]
    tvars_teacher_group3block0conv2 = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/group_3/unit_3_0/sub2/conv2/weights")][0]

    for var in tf.global_variables():
        if var.op.name.startswith("model/student_architecture/init"):
            print(session.run(var[0][0][0][0]))

    for var in tf.global_variables():
        if var.op.name.startswith("model/student_architecture/init/init_conv/weights"):
            tf.logging.info(var)
            var.assign(tvars_teacher_conv1.eval(session=session)).eval(session=session)

        if var.op.name.startswith("model/student_architecture/group_1/unit_1_0/sub1/conv1/weights"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group1block0conv1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_1/unit_1_0/sub2/conv2/weights"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group1block0conv2.eval(session=session)).eval(session=session)

        if var.op.name.startswith("model/student_architecture/group_2/unit_2_0/sub1/conv1/weights"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group2block0conv1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_2/unit_2_0/sub2/conv2/weights"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group2block0conv2.eval(session=session)).eval(session=session)

        if var.op.name.startswith("model/student_architecture/group_3/unit_3_0/sub1/conv1/weights"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group3block0conv1.eval(session=session)).eval(session=session)
        if var.op.name.startswith("model/student_architecture/group_3/unit_3_0/sub2/conv2/weights"):
            tf.logging.info(var)
            var.assign(tvars_teacher_group3block0conv2.eval(session=session)).eval(session=session)

    for var in tf.global_variables():
        if var.op.name.startswith("model/student_architecture/init"):
            print("after")
            print(session.run(var[0][0][0][0]))

def initialize_lastUnit_weights(session):
    tf.logging.info("Begin to initialize lastUnit weights.................................")
    tvars = tf.trainable_variables()
    tvars_teacher_lastweights = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/unit_last/FC/weights")][0]
    tvars_teacher_lastbiases = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/unit_last/FC/biases")][0]
    for var in tf.global_variables():
        if var.op.name.startswith("model/student_architecture/unit_last/FC/weights"):
          tf.logging.info(var)
          var.assign(tvars_teacher_lastweights.eval(session=session)).eval(session=session)

        if var.op.name.startswith("model/student_architecture/unit_last/FC/biases"):
          tf.logging.info(var)
          var.assign(tvars_teacher_lastbiases.eval(session=session)).eval(session=session)

def initialize_lastUnit_gammaBeta(session):
    tf.logging.info("Begin to initialize lastUnit gamma and beta.................................")
    tvars = tf.trainable_variables()
    tvars_teacher_lastgamma = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/unit_last/final_bn/gamma")][0]
    tvars_teacher_lastbeta = [var for var in tvars if var.op.name.startswith("model/teacher_architecture/unit_last/final_bn/beta")][0]
    for var in tf.global_variables():
        if var.op.name.startswith("model/student_architecture/unit_last/final_bn/gamma"):
          tf.logging.info(var)
          var.assign(tvars_teacher_lastgamma.eval(session=session)).eval(session=session)

        if var.op.name.startswith("model/student_architecture/unit_last/final_bn/beta"):
          tf.logging.info(var)
          var.assign(tvars_teacher_lastbeta.eval(session=session)).eval(session=session)



