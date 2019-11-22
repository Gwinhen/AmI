#!/usr/bin/env python
# coding=utf-8
import torch
import cv2
import numpy as np
import re


SKIP_LAYERS = [
    'relu1_1',
    'relu1_2',
    'relu2_1',
    'relu2_2',
    'relu3_1',
    'relu3_2',
    'relu3_3',
    'relu4_1',
    'relu4_2',
    'relu4_3',
    'relu5_1',
    'relu5_2',
    'relu5_3',
    'relu6',
    'dropout6',
    'relu7',
    'dropout7',
    'fc8',
]


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
        return torch.from_numpy(np.load(img_path)).unsqueeze(0)
    else:
        return torch.from_numpy(get_vgg_data(img_path))


def load_neuron_set_lists():
    res = []
    attributes = ['leye', 'reye', 'mouth', 'nose', ]
    for attri in attributes:
        filename = f'./ami_data/{attri}_neurons.txt'
        tmp = {}
        with open(filename) as in_file:
            for line in in_file:
                line = line.strip()
                if line:
                    name, line = line.split('->')
                    tmp[name] = list(map(int, line.split(','))) if line else []
        res.append(tmp)
    assert len(res) == len(attributes)
    return res


def get_identity(img_name, names):
    indices = [i.start() for i in re.finditer('_', img_name)]
    name = img_name[:indices[len(indices)-5]]
    if name in names:
        return names.index(name)


def read_list(f):
    l = []
    with open(f) as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                l.append(line)
    return l
