#!/usr/bin/env python
# coding=utf-8
import torch
import cv2
import numpy as np
import re


SKIP_LAYERS = [
    'conv1_1',
    'conv1_2',
    'conv2_1',
    'conv2_2',
    'conv3_1',
    'conv3_2',
    'conv3_3',
    'conv4_1',
    'conv4_2',
    'conv4_3',
    'conv5_1',
    'conv5_2',
    'conv5_3',
    'fc6',
    'dropout6',
    'fc7',
    'dropout7',
    'fc8',
]


def get_vgg_data(img_path, is_rgb):
    # averageImg = [129.1863, 104.7624, 93.5940]
    averageImg = [129.186279296875, 104.76238250732422, 93.59396362304688]
    img = cv2.imread(img_path)
    if img.shape[0] != 224:
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if is_rgb:
        data = np.float32(np.rollaxis(img, 2)) # RGB
        data[0] -= averageImg[0]
        data[1] -= averageImg[1]
        data[2] -= averageImg[2]
    else:
        data = np.float32(np.rollaxis(img, 2)[::-1]) # BGR
        data[0] -= averageImg[2]
        data[1] -= averageImg[1]
        data[2] -= averageImg[0]
    return np.array([data])


def get_data(img_path, is_rgb=True):
    if '.npy' in img_path:
        img = np.load(img_path) # BGR
        if is_rgb:
            img = np.ascontiguousarray(img[::-1]) # RGB
        return torch.from_numpy(img).unsqueeze(0)
    else:
        return torch.from_numpy(get_vgg_data(img_path, is_rgb))


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


def show_pth(pth_path):
    od = torch.load(pth_path)
    for k, v in od.items():
        print(f'{k}:\t{v.shape}')

def align_pth_keys(pth_path, keys):
    new_pth = OrderedDict()
    old_pth = torch.load(pth_path)
    for i, v in enumerate(old_pth.values()):
        new_pth[keys[i]] = v
    return new_pth
