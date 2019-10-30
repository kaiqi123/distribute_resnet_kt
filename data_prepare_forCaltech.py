import pickle as cPickle
#import cPickle
import os
import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_SIZE = 32

def save_img2(imgs, fileDir):
    img0 = imgs[0]
    img1 = imgs[1]
    img2 = imgs[2]
    i0 = Image.fromarray(img0).convert('L')
    i1 = Image.fromarray(img1).convert('L')
    i2 = Image.fromarray(img2).convert('L')
    img = Image.merge("RGB", (i0, i1, i2))
    img.save(fileDir, "png")

def load_caltech101(filename):
    with open(filename, 'rb')as f:
        datadict = cPickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        assert X.shape[0] == len(Y)
        X = X.reshape(X.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
        Y = np.array(Y)
        return X, Y

def read_caltechPictures(openFile):
    data = []
    labels = []
    data_dict = {}
    labelName = -1
    for subdir, dirs, files in os.walk(openFile):
        #print(subdir, dirs, files)
        print("label: "+str(labelName))
        for file in files:
            img = Image.open(subdir+"/"+file)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            img_ndarray = np.array(img)
            if img_ndarray.ndim == 2:
                img_ndarray = np.stack((img_ndarray,) * 3, axis=-1)
            img_ndarray = img_ndarray.transpose((2,0,1))
            data.append(img_ndarray)
            labels.append(labelName)
        labelName = labelName + 1
    assert len(data)==len(labels)
    data = np.array(data).reshape(len(data), 3*IMAGE_SIZE*IMAGE_SIZE)
    data_dict["data"] = data
    data_dict["labels"] = labels
    print(data.shape)
    return data_dict

def create_train_test_examples_saveToPkl(data_dict,writeFileDir):
    data   = list(data_dict["data"])
    labels = data_dict["labels"]
    print(labels)
    print(len(data))
    train_labels = []
    train_data = []
    test_labels = []
    test_data = []
    begin_index = 0
    for i in range(102):
        num_oneCategory = labels.count(i)
        num_train = int(num_oneCategory*0.8)

        train_labels_oneCategory = labels[begin_index:begin_index+num_train]
        test_labels_oneCategory = labels[begin_index+num_train:begin_index+num_oneCategory]
        train_data_oneCategory = data[begin_index:begin_index+num_train]
        test_data_oneCategory = data[begin_index+num_train:begin_index+num_oneCategory]

        train_labels = train_labels + train_labels_oneCategory
        test_labels = test_labels + test_labels_oneCategory
        train_data = train_data + train_data_oneCategory
        test_data = test_data + test_data_oneCategory
        begin_index = begin_index + num_oneCategory
        print("num_oneCategory: "+str(num_oneCategory))
        print("num_train: "+str(num_train))
    assert len(train_data)==len(train_labels)
    assert len(test_data)==len(test_labels)
    print(train_labels)
    print(len(train_data))
    print(test_labels)
    print(len(test_data))

    test_dict={}
    test_dict["data"] = np.array(test_data)
    test_dict["labels"] = test_labels
    file_writer_test = open(writeFileDir+"/test.pkl", 'wb')
    cPickle.dump(test_dict, file_writer_test)
    file_writer_test.close()

    begin_index = 0
    num = int(len(train_data) / 5)
    for i in range(1,6):
        train_dict = {}
        train_dict["data"] = np.array(train_data[begin_index:begin_index+num])
        train_dict["labels"] = train_labels[begin_index:begin_index+num]
        file_writer_train = open(writeFileDir + "/train_"+str(i)+".pkl", 'wb')
        cPickle.dump(train_dict, file_writer_train)
        file_writer_train.close()
        begin_index = begin_index + num

def unpickle(f):
  print('loading file: {}'.format(f))
  fo = tf.gfile.Open(f, 'r')
  d = cPickle.load(fo)
  fo.close()
  return d

def read_pklData(data_path, datafiles):
    all_data = []
    all_labels = []
    for file_num, f in enumerate(datafiles):
        print(file_num, f)
        d = unpickle(os.path.join(data_path, f))
        data = d['data']
        labels = d['labels']
        all_data = all_data + list(data)
        all_labels = all_labels + labels
        print(data.shape)
        print(len(labels))
    all_data = np.array(all_data)
    print(type(all_data), all_data.shape)
    print(type(all_labels), len(all_labels))

#data_dict = read_caltechPictures("./data_prepare/test")
data_dict = read_caltechPictures("./101_ObjectCategories")
create_train_test_examples_saveToPkl(data_dict, "./caltech101-batches-py")

datafiles = ['train_1.pkl', 'train_2.pkl', 'train_3.pkl', 'train_4.pkl', 'train_5.pkl', 'test.pkl']
read_pklData("./caltech101-batches-py", datafiles)

"""
for i in range(1,6):
    imgX, imgY = load_caltech101("./caltech101-batches-py/train_"+str(i)+".pkl")
    for j in range(imgX.shape[0]):
        save_img2(imgX[j], "./data_prepare/test_output/train_"+str(i)+"_Label"+str(imgY[j])+".jpg")

datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
read_pklData("./cifar-10-batches-py", datafiles)
"""

"""
all: 9144 (origin 9145, less a training example)
training: 1456*5=7280
testing: 1864
"""
