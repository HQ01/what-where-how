from pdb import set_trace as st
import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utilities import *


class MLP(nn.Module):
    def __init__(self, in_features, out_features, features, nonlinear):
        super(MLP, self).__init__()
        shapes = zip((in_features, ) + features, features + (out_features, ))
        self._linear_list = nn.ModuleList()
        for shape in shapes:
            self._linear_list.append(nn.Linear(*shape))
        self._nonlinear = nonlinear

    def forward(self, data):
        if len(data.size()) != 2:
            data = data.view(data.size()[0], -1)
        for index, linear in enumerate(self._linear_list):
            data = linear(data)
            data = self._nonlinear(data)
        if index < len(self._linear_list) - 1:
            data = self._linear_list[-1](data)
        return data


class Network0(nn.Module):
    def __init__(self, feature_extractor, n_features, size, T, penalty=0):
        super(Network0, self).__init__()
        self._feature_extractor = feature_extractor
        self._mask_generator = nn.Linear(n_features, 4)
        self._classifier = nn.Linear(n_features, 10)
        self._gaussian_mask = GaussianMask(*size)
        self._T = T
        self._penalty = penalty

    def cuda(self):
        super(Network0, self).cuda()
        self._gaussian_mask.cuda()

    def forward(self, data):
        internal = {'data': data, 'mask_list': [], 'stat_list': []}

        N = data.size()[0]
        mx = (th.rand(N, 1) - 0.5) / 2
        my = (th.rand(N, 1) - 0.5) / 2
        sx, sy = th.rand(N, 1), th.rand(N, 1)
        # sx, sy = th.ones(N, 1) * 0.5, th.ones(N, 1) * 0.5
        if data.is_cuda:
            mx, my, sx, sy = map(lambda t: t.cuda(), (mx, my, sx, sy))
        mx, my, sx, sy = map(Variable, (mx, my, sx, sy))
        mask = self._gaussian_mask(mx, my, sx, sy)
        prediction_list = []
        for _ in range(self._T):
            internal['mask_list'].append(mask)
            internal['stat_list'].append((mx, my, sx, sy))

            masked = mask * data
            features = self._feature_extractor(masked)

            stats = self._mask_generator(features)
            mx, my, sx, sy = th.chunk(stats, 4, 1)
            mx, my = th.tanh(mx), th.tanh(my)
            sx, sy = th.exp(sx), th.exp(sy)
            sx = th.clamp(sx, min=0.1)
            sy = th.clamp(sy, min=0.1)

            mask = self._gaussian_mask(mx, my, sx, sy)

            category = self._classifier(features)
            prediction_list.append(category)

        return prediction_list, internal

    def loss(self, prediction_list, labels, internal):
        ce_list = cross_entropy(prediction_list, labels)
        ce = sum(ce_list) / self._T

        if self._T > 1:
            penalty = sum(
                th.mean(th.abs(sx)) + th.mean(th.abs(sy))
                for _, _, sx, sy in internal['stat_list'][1:]) / (self._T - 1)
        else:
            penalty = 0

        value = ce + self._penalty * penalty
        return value, ce_list
