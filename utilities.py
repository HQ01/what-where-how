from __future__ import division
from pdb import set_trace as st
from argparse import Action
import math
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *
import joblib


class GaussianMask(nn.Module):
    def __init__(self, width, height):
        super(GaussianMask, self).__init__()
        self._width, self._height = width, height
        x = tuple(th.linspace(-1.0, 1.0, width) for _ in range(height))
        x = tuple(th.unsqueeze(t, 0) for t in x)
        y = tuple(th.linspace(-1.0, 1.0, height) for _ in range(width))
        y = tuple(th.unsqueeze(t, 1) for t in y)
        self._x = th.unsqueeze(th.cat(x, 0), 0)
        self._y = th.unsqueeze(th.cat(y, 1), 0)
        self._x, self._y = Variable(self._x), Variable(self._y)

    def cuda(self):
        self._x, self._y = self._x.cuda(), self._y.cuda()

    def forward(self, mx, my, sx, sy):
        """
        Parameters
        ----------
        mx, my : torch.autograd.Variable of shape (N, 1)
            Mean of X and Y axes respectively.
        sx, sy : torch.autograd.Variable of shape (N, 1)
            Standard deviation of X and Y axes respectively.
        """

        N = mx.size()[0]
        x, y = self._x, self._y
        x = x.expand(N, self._width, self._height)
        y = y.expand(N, self._width, self._height)

        def expand(stat):
            stat = th.unsqueeze(stat, 2)
            stat = stat.expand(N, self._width, self._height)
            return stat
        mx, my = expand(mx), expand(my)
        sx, sy = expand(sx), expand(sy)

        z = ((x - mx) / sx)**2 + ((y - my) / sy)**2
        mask = 1 / (2 * math.pi * sx * sy) * th.exp(-z / 2)
        return mask

def create_mnist_loaders(mnist_path, batch_size):
    partition_tuple = 'train', 'validate', 'test'
    size, data_tuple = joblib.load(mnist_path)
    data_tuple = tuple(map(th.from_numpy, d) for d in data_tuple)
    data_tuple = tuple(TensorDataset(*d) for d in data_tuple)
    data_dict = dict(zip(partition_tuple, data_tuple))

    loader_dict = {}
    kwargs = {'batch_size': batch_size, 'num_workers': 4, 'drop_last': True}
    for key, value in data_dict.items():
        kwargs['shuffle'] = (key == 'train')
        loader_dict[key] = DataLoader(value, **kwargs)

    return loader_dict, size

class parse_nonlinear(Action):
    def __call__(self, parser, namespace, values, option_string):
        if hasattr(F, values):
            values = getattr(F, values)
        else:
            values = None
        setattr(namespace, self.dest, values)

def partition(delimiter, type):
    class action(Action):
        def __call__(self, parser, namespace, values, option_string):
            values = tuple(values.split(delimiter))
            values = map(type, values)
            setattr(namespace, self.dest, values)

    return action

def accuracy(p_list, labels):
    def accuracy(p):
        indicator = th.max(p, 1)[1].long() != labels
        n_errors = th.sum(indicator.int()).data[0]
        n_samples = p.size()[0]
        accuracy = 1 - n_errors / n_samples
        return accuracy

    a_tuple = map(accuracy, p_list)
    return a_tuple

def cross_entropy(prediction_list, label):
    loss_list = []
    for prediction in prediction_list:
        ce = F.cross_entropy(prediction, label)
        loss_list.append(ce)

    return loss_list

def weighted_cross_entropy(p_list, label, g):
    loss_list = []
    for i, prediction in enumerate(p_list):
        ce = F.cross_entropy(prediction, label)
        gamma = g(i, len(p_list))
        loss_list.append(ce * gamma)

    return loss_list
