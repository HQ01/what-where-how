from pdb import set_trace as st
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utilities import GaussianMask


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 features,
                 nonlinear):
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
    def __init__(self, feature_extractor, n_features, size, T):
        super(Network0, self).__init__()
        self._feature_extractor = feature_extractor
        self._mask_generator = nn.Linear(n_features, 4)
        self._classifier = nn.Linear(n_features, 10)
        self._gaussian_mask = GaussianMask(*size)
        self._T = T

    def cuda(self):
        super(Network0, self).cuda()
        self._gaussian_mask.cuda()

    def forward(self, data):
        N = data.size()[0]
        mx = (th.rand(N, 1) - 0.5) / 2
        my = (th.rand(N, 1) - 0.5) / 2
        sx, sy = th.rand(N, 1), th.rand(N, 1)
        if data.is_cuda:
            mx, my, sx, sy = map(lambda t: t.cuda(), (mx, my, sx, sy))
        mx, my, sx, sy = map(Variable, (mx, my, sx, sy))
        mask = self._gaussian_mask(mx, my, sx, sy)
        prediction_list = []
        for _ in range(self._T):
            masked = mask * data
            features = self._feature_extractor(masked)

            stats = self._mask_generator(features)
            mx, my, sx, sy = th.chunk(stats, 4, 1)
            mx, my = th.tanh(mx), th.tanh(my)
            sx, sy = th.exp(sx), th.exp(sy)
            mask = self._gaussian_mask(mx, my, sx, sy)

            category = self._classifier(features)
            prediction_list.append(category)

        return prediction_list
