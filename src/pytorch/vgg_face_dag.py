#!/usr/bin/env python
# coding=utf-8
"""
Original Vgg_face_dag codes are downloaded from http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.py
Functions related to hooks are added by me
"""
import torch
import torch.nn as nn
from collections import OrderedDict


class Hook():
    def __init__(self, name, module, weaken_param, strengthen_param0, strengthen_param1, ami_data=None, return_tensor=False):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.name = name
        self.output = None
        self.ami_data = ami_data
        self.weaken_neurons = None
        self.strengthen_neurons = []
        self.attri_neurons = []
        self.processed = False
        self.return_tensor = return_tensor
        self.weaken_param = weaken_param
        self.strengthen_param0 = strengthen_param0
        self.strengthen_param1 = strengthen_param1

    def hook_fn(self, module, input, output):
        # self.input = input.detach().clone().cpu().numpy()
        # output shape: batch x neurons x ...
        if self.ami_data is not None:
            if (not self.processed) and (not self.strengthen_neurons):
                self.weaken_neurons = torch.ones(output.shape[1], device=output.device)
                for n_l in self.ami_data:
                    n_l = n_l[self.name]
                    self.attri_neurons.extend(n_l)
                    if not n_l: continue
                    tmp_n = torch.zeros(output.shape[1], device=output.device)
                    tmp_n[n_l] = 1
                    self.strengthen_neurons.append(tmp_n)
                    self.weaken_neurons *= (1-tmp_n)
                self.processed = True
            if not self.attri_neurons or not self.strengthen_neurons:
                return output
            new_output = neuron_AmI(output, self.weaken_neurons, self.strengthen_neurons,
                                    self.attri_neurons, self.name, self.weaken_param,
                                    self.strengthen_param0, self.strengthen_param1)
            return new_output
        else:
            if self.return_tensor:
                self.output = output.detach().clone()
            else:
                self.output = output.detach().clone().cpu().squeeze(0).numpy()

    def close(self):
        self.hook.remove()


class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)
        self.my_hooks = {}

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)

        if len(self.my_hooks) > 0 and list(self.my_hooks.values())[0].ami_data is None:
            res = OrderedDict()
            for n, h in self.my_hooks.items():
                if h.output is None:
                    print(f'{n}.output is None')
                res[n] = h.output
            return res

        return x38

    def set_ami_params(self, ami_weaken_parameter, ami_strengthen_parameter0, ami_strengthen_parameter1):
        self.weaken_param = ami_weaken_parameter
        self.strengthen_param0 = ami_strengthen_parameter0
        self.strengthen_param1 = ami_strengthen_parameter1

    def register_my_hook(self, skip_layers=[], ami_data=None, return_tensor=False):
        self.my_hooks = OrderedDict()
        if ami_data is None:
            # set fake ami params to avoid errors
            self.set_ami_params(0,0,0)
        for name, module in self.named_modules():
            if name not in skip_layers:
                if len(list(module.children())) == 0:
                    print(f'register hook for {name}')
                    self.my_hooks[name] = Hook(name, module, self.weaken_param,
                                               self.strengthen_param0, self.strengthen_param1,
                                               ami_data, return_tensor)


def neuron_AmI(x, weaken_neurons, strengthen_neurons, attri_neurons, layer_i, weaken_param, strengthen_param0, strengthen_param1):
    x = x.squeeze(0)
    if len(x.shape) == 1:
        data = x
    else:
        data = torch.sum(x, (1,2))
    attri_data = data[attri_neurons]
    attri_mean = attri_data.mean()
    attri_std = attri_data.std(unbiased=False)
    attri_min = attri_data.min()

    deviation = torch.zeros(x.shape[0], device=weaken_neurons.device)
    if attri_std != 0:
        deviation = torch.max(deviation, (data-attri_mean)/attri_std)
    wkn = weaken(deviation, weaken_param) * weaken_neurons

    stn = 1.
    for st_n in strengthen_neurons:
        deviation = torch.zeros(x.shape[0], device=weaken_neurons.device)
        if attri_std != 0:
            deviation = torch.abs(data-attri_min) / attri_std
        t_stn = strengthen(deviation, strengthen_param0, strengthen_param1) * st_n + (1 - st_n)
        stn *= t_stn
    stn *= (1.-weaken_neurons)

    stn += wkn
    if len(x.shape) == 3:
        stn.unsqueeze_(-1)
        stn.unsqueeze_(-1)
    return (x*stn).unsqueeze(0)


def strengthen(x, strengthen_param0, strengthen_param1):
    return strengthen_param0 - torch.exp(-x/strengthen_param1)


def weaken(x, weaken_param):
    return torch.exp(-x/weaken_param)


def vgg_face_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
