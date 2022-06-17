
import math
import torch
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import InstanceNorm1d as InstanceNorm
from torch.nn import *


from .activations import Cosine, Sine, SquarePlus, SoftPlusZero


def create_sequential_linear_layer(layers_size, act_name, norm_name, drop_prob=0.0, bias=True, last_act=True, act_params={}):
    ## create a sequential layer containing Linear + activation
    def create_layer(in_feat, out_feat, bias):
        return Linear(in_feat, out_feat, bias=bias)

    layers  = [(in_feat, out_feat) for in_feat, out_feat in zip(layers_size[:-1], layers_size[1:])]
    return create_sequential(create_layer, layers, act_name, norm_name, drop_prob, bias=bias, last_act=last_act, act_params=act_params)


def create_sequential_conv_layer(layers_size, act_name, norm_name, drop_prob=0.0, last_act=True):
    ## create sequential layer containing Conv2d + activation
    def create_layer(in_feat, out_feat, bias):
        return Conv2d(in_feat, out_feat[0], kernel_size=(1, out_feat[1]))
    return create_sequential(create_layer, layers_size, act_name, norm_name, drop_prob, bias=True, last_act=last_act)


def create_sequential(LayerClass, layers, act_name, norm_name, drop_prob, bias=True, last_act=True, act_params={}):

    act_fun    = get_activation_function(act_name)
    norm_layer = get_normalization_layer(norm_name)
    #################################
    # Create Sequential dynamically #
    #################################
    modules = []

    ## Generate a Sequential model based on the number of units specified in `model_structure`
    for (in_feat, out_feat) in layers:
        modules.append(LayerClass(in_feat, out_feat, bias))
        if norm_name is not None:
            modules.append(norm_layer(num_features=out_feat))
        if drop_prob > 0.0:
            modules.append(Dropout(p=drop_prob))
        modules.append(act_fun(**act_params))

    ## remote last activation
    if not last_act:
        modules = modules[:-1]
        if drop_prob > 0.0:
            modules = modules[:-1]  # remove also dropout
        if norm_name is not None:
            modules = modules[:-1] # remove normalization

    return Sequential(*modules)


def get_init_fun(init_fun):

    if init_fun == 'xavier':
        init = torch.nn.init.xavier_normal_
    elif init_fun == 'ortho':
        init = torch.nn.init.orthogonal_
    elif init_fun == 'kaiming':
        init = torch.nn.init.kaiming_normal_
    elif 'siren' in init_fun:
        def init(weight):
            n = float(weight.size(-1)) # num input features
            b = torch.sqrt(torch.FloatTensor([6])/n).item()
            a = -b
            torch.nn.init.uniform_(weight, a, b)
    elif init_fun == 'selu':
        def init(weight):
            torch.nn.init.normal_(weight, 0.0, 0.5 / math.sqrt(weight.numel()))
    elif init_fun == 'uniform':
        def init(weight):
            torch.nn.init.uniform_(weight, -1.0, 1.0)
    elif init_fun == 'normal':
        def init(weight):
            torch.nn.init.uniform_(weight, 0.0, 1.0)

    def initialize(m):
        if type(m) == Conv1d or type(m) == Conv2d or type(m) == Linear or 'Linear' in type(m).__name__:
            init(m.weight)

    return initialize


def get_activation_function(act_name):
    act = globals()[act_name]

    return act


def get_normalization_layer(norm_name):

    if norm_name == 'batch':
        norm = BatchNorm
    elif norm_name == 'instance':
        # norm = InstanceNorm
        norm = TransposeInstanceNorm
    elif norm_name == 'layer':
        norm = LayerNorm
    elif norm_name == 'local':
        norm = LocalResponseNorm
    else:
        norm = Identity
    return norm


class TransposeInstanceNorm(torch.nn.Module):

    def __init__(self, num_features, affine=False):
        super().__init__()
        self.norm = InstanceNorm(num_features, affine=affine)

    def forward(self, x):
        return self.norm(x.transpose(1,2)).transpose(1,2)
