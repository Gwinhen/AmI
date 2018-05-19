import cv2
import numpy as np
import re


def read_list(f):
    l = []
    for line in open(f, 'rb'):
        l.append(line.strip())
    return l


def get_identity(img_name, names):
    indices = [i.start() for i in re.finditer('_', img_name)]
    name = img_name[:indices[len(indices)-5]]
    if name in names:
        return names.index(name)


def get_vgg_data(img_path):
    averageImg = [129.1863, 104.7624, 93.5940]
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
        return np.load(img_path)
    else:
        return get_vgg_data(img_path)


def get_prob(net, img_path):
    net.blobs['data'].data[...] = get_data(img_path)
    net.forward()
    return net.blobs['prob'].data[0].copy()


def get_layers(net):
    layers = []
    for layer in net.blobs:
        layers.append(layer)
    return layers


def get_layer_size(net, layer):
    return len(net.params[layer][0].data)