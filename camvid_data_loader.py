from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os

# Copy the data to this dir here in the SegNet project /CamVid from here:
# https://github.com/alexgkendall/SegNet-Tutorial
DataPath = './CamVid/'
# data_shape = 360*480
data_shape = 224 * 224


def load_data(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        # 示例数据
        # /SegNet/CamVid/train/0001TP_006690.png /SegNet/CamVid/trainannot/0001TP_006690.png
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        # this load cropped images

        # 关于numpy.rollaxis()的说明
        # axes = list(range(0, n))
        # axes.remove(axis)
        # axes.insert(start, axis)
        # return a.transpose(axes)
        # 把元素从一个位置移动到另一个位置

        # 关于os.getcwd()的说明
        # os.getcwd()
        # '/Users/demons/Documents/One-Hundred-Layers-Tiramisu'

        # txt[i][0][7:]的示例数据
        # /CamVid/train/0001TP_006690.png

        # image.shape = (360, 480, 3)
        # 360 - 136 = 224
        # 480 - 256 = 224
        # ...[136:, 256:]通过裁剪得到右下角的图片

        # arr = np.ones([224, 224, 3])
        # np.rollaxis(arr, 2).shape
        # (3, 224, 224)
        # 如果我们使用tensorflow作为backend，可能不需要改变图像的维度顺序

        data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])[136:, 256:]), 0))
        label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[136:, 256:][:, :, 0], 224, 224))
    return np.array(data), np.array(label)



train_data, train_label = load_data("train")
train_label = np.reshape(train_label, (train_data.shape[0], data_shape, 12))

test_data, test_label = load_data("test")
test_label = np.reshape(test_label, (test_data.shape[0], data_shape, 12))

val_data, val_label = load_data("val")
val_label = np.reshape(val_label, (val_data.shape[0], data_shape, 12))


np.save("data/train_data", train_data)
np.save("data/train_label", train_label)

np.save("data/test_data", test_data)
np.save("data/test_label", test_label)

np.save("data/val_data", val_data)
np.save("data/val_label", val_label)

# FYI they are:
# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road_marking = [255,69,0]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]
