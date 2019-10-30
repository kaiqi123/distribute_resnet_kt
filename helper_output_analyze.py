
import tensorflow as tf
import numpy as np

"""
fc Tensor("model_1/student_architecture/unit_last/FC/xw_plus_b:0", shape=(128, 10), dtype=float32, device=/device:GPU:0)
group1_block0_sub1_relu Tensor("model_1/student_architecture/group_1/unit_1_0/shared_activation/Relu:0", shape=(128, 32, 32, 16), dtype=float32, device=/device:GPU:0)
group1_block0_sub2_relu Tensor("model_1/student_architecture/group_1/unit_1_0/sub2/Relu:0", shape=(128, 32, 32, 160), dtype=float32, device=/device:GPU:0)
group2_block0_sub1_relu Tensor("model_1/student_architecture/group_2/unit_2_0/residual_only_activation/Relu:0", shape=(128, 32, 32, 160), dtype=float32, device=/device:GPU:0)
group2_block0_sub2_relu Tensor("model_1/student_architecture/group_2/unit_2_0/sub2/Relu:0", shape=(128, 16, 16, 320), dtype=float32, device=/device:GPU:0)
group3_block0_sub1_relu Tensor("model_1/student_architecture/group_3/unit_3_0/residual_only_activation/Relu:0", shape=(128, 16, 16, 320), dtype=float32, device=/device:GPU:0)
group3_block0_sub2_relu Tensor("model_1/student_architecture/group_3/unit_3_0/sub2/Relu:0", shape=(128, 8, 8, 640), dtype=float32, device=/device:GPU:0)
unit_last_relu Tensor("model_1/student_architecture/unit_last/Relu:0", shape=(128, 8, 8, 640), dtype=float32, device=/device:GPU:0)
"""

def return_output_list(output_dict):
    output_list = [output_dict["group1_block0_sub1_relu"], output_dict["group1_block0_sub2_relu"],
                   output_dict["group2_block0_sub1_relu"], output_dict["group2_block0_sub2_relu"],
                   output_dict["group3_block0_sub1_relu"], output_dict["group3_block0_sub2_relu"],
                   output_dict["unit_last_relu"], output_dict["fc"]]
    return output_list

def count_filter0_num_fcLayers(output, about0num):
    #print(output.shape)
    filter_count = []
    for i in range(output.shape[0]):
        #print(output[i].shape)
        count0Num_perImg = 0
        for j in range(output.shape[1]):
            if output[i][j] <= about0num:
                count0Num_perImg = count0Num_perImg + 1
        filter_count.append(count0Num_perImg)
        #print(np.max(output[i]), np.min(output[i]))
    return filter_count

def count_filter0_num_convLayers(output, perNum):
    filter_count = []
    for i in range(output.shape[0]):
        img = output[i]
        img = img.transpose(2, 0, 1)
        count = 0
        for j in range(img.shape[0]):

            # count number filters whose 90% output_wrn are 0
            num_sum = float(img[j].shape[0] * img[j].shape[1])
            count0_perFIlter = (num_sum - np.count_nonzero(img[j])) / num_sum
            if count0_perFIlter >= perNum:
                count = count + 1

        filter_count.append(count)
    return filter_count

def run_output_list_perIteration(session, model, train_images, train_labels, avg_num0filters_perEpoch_dict):
    output_list = session.run(model.output_list,
                              feed_dict={
                                  model.images: train_images,
                                  model.labels: train_labels,
                              })

    if model.type == "independent_student":
        layer_names = ["group1_block0_sub1_relu", "group1_block0_sub2_relu",
                       "group2_block0_sub1_relu", "group2_block0_sub2_relu",
                       "group3_block0_sub1_relu", "group3_block0_sub2_relu",
                       "unit_last_relu", "fc"]

    assert len(layer_names) == len(output_list)
    for i in range(len(output_list)):
        #print(layer_names[i])
        if "fc" not in layer_names[i]:
            avg_num0filters_perBatch = np.mean(count_filter0_num_convLayers(output_list[i], perNum=1.0))
        else:
            avg_num0filters_perBatch = np.mean(count_filter0_num_fcLayers(output_list[i], about0num=1.0))

        if layer_names[i] in avg_num0filters_perEpoch_dict.keys():
            avg_num0filters_perEpoch_dict[layer_names[i]].append(avg_num0filters_perBatch)
            #print(avg_num0filters_perBatch)

    return avg_num0filters_perEpoch_dict


















