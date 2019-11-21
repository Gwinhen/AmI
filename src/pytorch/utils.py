#!/usr/bin/env python
# coding=utf-8
import torch
import cv2
import numpy as np

def get_vgg_data(img_path):
    # averageImg = [129.1863, 104.7624, 93.5940]
    averageImg = [129.186279296875, 104.76238250732422, 93.59396362304688]
    img = cv2.imread(img_path)
    if img.shape[0] != 224:
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = np.float32(np.rollaxis(img, 2)[::-1])
    data[0] -= averageImg[2]
    data[1] -= averageImg[1]
    data[2] -= averageImg[0]
    return np.array([data])


def get_data(img_path):
    if '.npy' in img_path:
        return torch.from_numpy(np.load(img_path))
    else:
        return torch.from_numpy(get_vgg_data(img_path))
