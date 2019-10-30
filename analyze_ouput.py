import numpy as np
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import linalg as LA
from os import listdir
import logging
import json
from scipy import stats
import os

def l2_norm_one_batch(data):
    print("original data.shape: " + str(data.shape))
    data = data.transpose(0, 3, 1, 2)
    norm = LA.norm(data, axis=(0))
    data_norm = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for i in range(data.shape[0]):
        data_norm[i] = np.true_divide(data[i], norm)
        #data_norm[i] = data[i] / norm
        data_norm[i][~np.isfinite(data_norm[i])] = 0.
    data_norm_new = data_norm.transpose(0, 2, 3, 1)
    print("data.shape: "+ str(data.shape))
    #print(data)
    print("norm.shape: " + str(norm.shape))
    #print(norm)
    print("data_norm.shape: " + str(data_norm.shape))
    #print(data_norm)
    print("data_norm_new.shape: " + str(data_norm_new.shape))
    return data_norm_new

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
    print(output.shape)
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

def loadOutputLog_calcuteMean(filename, searchName):
    logging.info(searchName)
    f = open(filename, "r")
    data = f.readlines()
    for i in range(len(data)):
        r = data[i].find(searchName)
        if r > -1:
            s = data[i + 1].split(":")[2]
            s = json.loads(s)
            break
    f.close()
    logging.info(s)
    logging.info("Mean: {}, Min: {}, Mode: {}.\n".format(np.mean(s), np.min(s), stats.mode(s)))

def readNpyFile_count0Filters_saveToLog_calculateMean(logName, readNpyPath, perOf0Filters):
    logging.basicConfig(filename=logName, level=logging.DEBUG)
    logging.FileHandler(logName, mode='w')

    searchNames = []
    for filename in sorted(os.listdir(readNpyPath)):
        logging.info(filename)
        output = np.load(readNpyPath + filename)
        if "conv" in readNpyPath:
            filter_count = count_filter0_num_convLayers(output, perOf0Filters)
        elif "fc" in readNpyPath:
            filter_count = count_filter0_num_fcLayers(output, perOf0Filters)
        logging.info(filter_count)
        searchNames.append(filename)

    #searchNames = ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]
    logging.info("\n\nBegin to compute mean, min, and mode of every layer's output")
    for searchName in searchNames:
        loadOutputLog_calcuteMean(logName, searchName)

def plot_images_afterRelu(dirNpy, inputNpy_origin, inputNpy_augment, saveName, flag_norm=False):
    output = np.load(dirNpy)
    print(output.shape)

    input_origin = np.load(inputNpy_origin)
    print(input_origin.shape)

    input_augment = np.load(inputNpy_augment)
    print(input_augment.shape)

    if flag_norm:
        output = l2_norm_one_batch(output)

    # for one image
    for i in range(100, 101):
        print("iteration: " + str(i))
        print("The shape of one image: " + str(output[i].shape))
        img = output[i]
        img = img.transpose(2, 0, 1)
        img_sum = np.sum(img, axis=0)
        fig = plt.figure(figsize=(11, 8))
        rows = 4
        columns = 6

        input_origin_img = input_origin[i]
        print(input_origin_img.shape, np.max(input_origin_img), type(input_origin_img))
        origin_figure=fig.add_subplot(rows, columns, 1)
        origin_figure.title.set_text('original image')
        origin_figure.axis('off')
        plt.imshow(input_origin_img)

        input_augment_img = input_augment[i]
        print(input_augment_img.shape, np.max(input_augment_img), type(input_augment_img))
        augment_figure = fig.add_subplot(rows, columns, 2)
        augment_figure.title.set_text('input image')
        augment_figure.axis('off')
        plt.imshow(input_augment_img)

        #fig.add_subplot(rows, columns, 3).axis('off')
        #plt.imshow(img_sum)

        for j in range(img.shape[0]):
            img_dim = img[j]
            if j>=0 and j<=3:
                sub_figure = fig.add_subplot(rows, columns, j+3)
            elif j>=4 and j<=7:
                sub_figure = fig.add_subplot(rows, columns, j+5)
            elif j>=8 and j<=11:
                sub_figure = fig.add_subplot(rows, columns, j+7)
            else:
                sub_figure = fig.add_subplot(rows, columns, j+9)
            sub_figure.title.set_text('filer_'+str(j+1))
            sub_figure.axis('off')
            plt.imshow(img_dim)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(saveName)
        plt.show()

# use python2
dirNpy = "output_d28w10/filters_npy/teacherGroup11_normFalse_iteration_0.npy"
inputNpy_origin = "output_d28w10/filters_npy/input_original_images_iteration_0.npy"
inputNpy_augment = "output_d28w10/filters_npy/input_augmented_images_iteration_0.npy"
saveName = "output_d28w10/images/visualization_of_activations.pdf"
plot_images_afterRelu(dirNpy, inputNpy_origin, inputNpy_augment, saveName, flag_norm=False)

readNpyPath = "output_vgg16/filters_npy_conv/"
logName = "output_vgg16/num_of_filter0/conv/log_0.1.log"
perOf0Filters = 0.1
#readNpyFile_count0Filters_saveToLog_calculateMean(logName, readNpyPath, perOf0Filters)

readNpyPath_fc = "output_vgg16/filters_npy_fc/"
logName_fc = "output_vgg16/num_of_filter0/fc/log_2.0.log"
about0num = 2.0
#readNpyFile_count0Filters_saveToLog_calculateMean(logName_fc, readNpyPath_fc, about0num)
