import tensorflow as tf
from numpy import linalg as LA
import numpy as np

import copy
import os


"""
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the variables to disk.
with tf.Session() as sess:
  sess.run(init_op) # initialize the variables

  inc_v1.op.run() # Do some work with the model.
  dec_v2.op.run() # Do some work with the model.

  save_path = saver.save(sess, "./temp/model.ckpt") # Save the variables to disk.
  print("Model saved in path: %s" % save_path)


tf.reset_default_graph()
# Create some variables.
#v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  #v1.initializer.run()
  saver.restore(sess, "./temp/model.ckpt")

  #print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())

with tf.Session() as sess:
  saver.restore(sess, "./temp/model.ckpt")
  print("v2 : %s" % v2.eval())
"""
"""
# chkp.print_tensors_in_checkpoint_file(teacher_checkpoint_path, tensor_name='', all_tensors=True)
# chkp.print_tensors_in_checkpoint_file(checkpoint_path, tensor_name='', all_tensors=True)
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key))
"""

"""
# l2 norm by numpy
def l2_norm_one_batch(data):
    data = data.transpose(0, 3, 1, 2)
    norm = LA.norm(data, axis=(0))
    # norm = np.zeros((data.shape[1], data.shape[2], data.shape[3]))
    data_norm = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for i in range(data.shape[0]):
        data_norm[i] = np.true_divide(data[i], norm)
        #data_norm[i] = data[i] / norm
        data_norm[i][~np.isfinite(data_norm[i])] = 0.
    data_norm_new = data_norm.transpose(0, 2, 3, 1)
    print(data.shape)
    print(data)
    print(norm.shape)
    print(norm)
    print(data_norm.shape)
    print(data_norm)
    print(data_norm_new.shape)
    print(data_norm_new)
    return data_norm_new

data = np.arange(120).reshape(4, 3, 5, 2)
data_norm_new = l2_norm_one_batch(data)
"""

"""
# l2 norm by tensorflow
data = np.arange(120).reshape(4, 3, 5, 2)
input_data = tf.convert_to_tensor(data,tf.float64)
output_wrn = tf.nn.l2_normalize(input_data, axis = 0)
print(output_wrn)
output_ts = tf.transpose(output_wrn, [3, 0, 1, 2])[0]
print(output_ts)
output_ts_va = tf.Variable(output_ts)
print(output_ts_va)
with tf.Session() as sess:
     print(sess.run(output_ts))
#     print(sess.run(output_wrn))
"""

"""
sess = tf.Session()
data = np.arange(120).reshape(4, 3, 5, 2)
input_data = tf.convert_to_tensor(data,tf.float64)
input_data = tf.transpose(input_data, [3, 0, 1, 2])
print(input_data)
#for i in range(input_data.get_shape()[0]):
count_teacher0Filter = 0
for i in range(input_data.get_shape()[0]):
    print(i)
    #oneImg = tf.transpose(input_data[i], [2, 0, 1])

    print(input_data[i])
    sum_oneFilter = tf.cast(tf.reduce_sum(input_data[i]), tf.float32)
    print(sess.run(sum_oneFilter))

    def true_pre():
        return count_teacher0Filter + 1

    def false_pre():
        return count_teacher0Filter + 0

    count_teacher0Filter = tf.cond(tf.equal(sum_oneFilter, tf.constant(3540, dtype=float)), true_pre, false_pre)
    print(sess.run(count_teacher0Filter))
    print(count_teacher0Filter)
"""

"""
filter_list = []
for i in range(2):
    data = np.arange(60).reshape(3, 5, 4)
    input_data = tf.convert_to_tensor(data,tf.float64)
    filter_list.append(input_data)

c = tf.stack(filter_list, 3)

data3 = np.arange(120).reshape(3, 5, 4, 2)
input_data3 = tf.convert_to_tensor(data3, tf.float64)

print(c)
print(input_data3)
"""

"""
# 4-dim matrix dimension
data = np.arange(120).reshape(4, 3, 5, 2)
input_data = tf.convert_to_tensor(data,tf.float64)
print(input_data)
output_ts = tf.transpose(input_data, [3, 0, 1, 2])[0]
print(output_ts)
output_ts2 = input_data[:,:,:,0]
print(output_ts2)
#with tf.Session() as sess:
#     print(sess.run(output_ts))
"""
def compute_mean_std(data):
    data = np.transpose(data, [0,3,1,2])
    data_mean = np.mean(data,axis=0)
    mean0 = np.mean(data_mean[0])
    mean1 = np.mean(data_mean[1])
    mean2 = np.mean(data_mean[2])
    mean = [mean0,mean1,mean2]

    data_std = np.std(data,axis=0)
    std0 = np.std(data_std[0])
    std1 = np.std(data_std[1])
    std2 = np.std(data_std[2])
    std = [std0,std1,std2]
    print(mean)
    print(std)
    return mean,std

def calculate_cosineSimilarity():
    data1 = tf.convert_to_tensor(np.arange(48).reshape(4, 2, 2, 3), dtype=tf.float64)
    data2 = tf.convert_to_tensor(np.arange(48).reshape(4, 2, 2, 3), dtype=tf.float64)
    product12 = tf.reduce_sum(tf.multiply(data1, data2))
    data1_sqrt = tf.sqrt(tf.reduce_sum(tf.square(data1)))
    data2_sqrt = tf.sqrt(tf.reduce_sum(tf.square(data2)))
    cosine = tf.divide(product12, tf.multiply(data1_sqrt, data2_sqrt))

    with tf.Session() as sess:
         print(sess.run(product12))
         print(sess.run(cosine))
         print(sess.run(x))
         print(sess.run(x_max))
         print(sess.run(x_index))

#calculate_cosineSimilarity()

#x = [tf.constant(1, name="a"), tf.constant(2, name="b")]
#x_max = tf.reduce_max(x, reduction_indices=[0], name="maxValue")
#x_index = tf.argmax(x, dimension=0, name="max_indexaaaa").eval()
#tensors_per_node = [node.values() for node in tf.get_default_graph().get_operations()]
#tensor_names = [tensor.name for tensors in tensors_per_node for tensor in tensors]
#print(x_index)
#for n in tf.get_default_graph().as_graph_def().node:
#    print('\n',n)
"""
graph = tf.get_default_graph()
list_of_tuples = [op.values()[0] for op in graph.get_operations()]
print(list_of_tuples)
with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(x_max))
    print(sess.run(x_index))
"""
def variable(name, shape, dtype, initializer, trainable):
  var = tf.get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      trainable=trainable)
  return var

def fc(inputs,
       num_units_out,
       scope=None,
       reuse=None):
    num_units_in = inputs.shape[1]
    weights_shape = [num_units_in, num_units_out]
    unif_init_range = 1.0 / (num_units_out)**(0.5)
    weights_initializer = tf.random_uniform_initializer(-unif_init_range, unif_init_range)
    weights = variable(
        name='weights',
        shape=weights_shape,
        dtype=tf.float32,
        initializer=weights_initializer,
        trainable=True)
    bias_initializer = tf.constant_initializer(0.0)
    biases = variable(
        name='biases',
        shape=[num_units_out,],
        dtype=tf.float32,
        initializer=bias_initializer,
        trainable=True)
    outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    return outputs

def zero_pad(inputs, in_filter, out_filter):
  outputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],[(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
  return outputs

x = tf.convert_to_tensor(np.arange(72).reshape(2,3,3,4), dtype=tf.float32)
y = zero_pad(x, 4, 8)
print(x)
print(y)

with tf.Session() as sess:
    xv = sess.run(x)
    print(xv)
    yv = sess.run(y)
    print(yv)

