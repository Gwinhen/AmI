#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.functional import interpolate


class AmIModel(nn.Module):

    def __init__(self, base, ami_weaken_parameter=0, ami_strengthen_parameter0=0, ami_strengthen_parameter1=0):
        super(AmIModel, self).__init__()
        self.base = base
        self.my_hooks = {}
        self.weaken_param = ami_weaken_parameter
        self.strengthen_param0 = ami_strengthen_parameter0
        self.strengthen_param1 = ami_strengthen_parameter1

    def set_ami_params(self, ami_weaken_parameter, ami_strengthen_parameter0, ami_strengthen_parameter1):
        self.weaken_param = ami_weaken_parameter
        self.strengthen_param0 = ami_strengthen_parameter0
        self.strengthen_param1 = ami_strengthen_parameter1

    def register_my_hook(self, skip_layers=[], ami_data=None, return_tensor=False):
        self.my_hooks = OrderedDict()
        for name, module in self.base.named_modules():
            if name not in skip_layers:
                if len(list(module.children())) == 0:
                    print(f'register hook for {name}')
                    self.my_hooks[name] = Hook(name, module, self.weaken_param,
                                               self.strengthen_param0, self.strengthen_param1,
                                               ami_data, return_tensor)
    def remove_my_hook(self):
        for name, hook in self.my_hooks.items():
            print(f'remove hook for {name}')
            hook.close()
        self.my_hooks = {}

    def show_layers(self):
        for name, module in self.base.named_modules():
            if len(list(module.children())) == 0:
                print(f'{name}:\t{module}')

    def get_activation_values(self):
        if len(self.my_hooks) > 0 and list(self.my_hooks.values())[0].ami_data is None:
            res = OrderedDict()
            for n, h in self.my_hooks.items():
                if h.output is None:
                    print(f'{n}.output is None')
                res[n] = h.output
            return res

    def forward(self, x0):
        return self.base(x0)


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
            if 'pool3' in self.name:
                t_h, t_w = output.shape[2:]
                tmp = output.clone()[...,2:t_h-2, 2:t_w-2]
                tmp = interpolate(tmp, size=(t_h, t_w), mode='bilinear', align_corners=False)
                output = tmp

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


