import torch.nn as nn
import torch.nn.functional as F
from utilities import GaussianMask


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 features,
                 linear=nn.Linear,
                 nonlinear=F.relu):
        super(MLP, self).__init__()
        shapes = zip((in_features, ) + features, features + (out_features, ))
        self._linear_list = nn.ModuleList()
        for shape in shapes:
            self._linear_list.append(linear(*shape))
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
    def __init__(self, feature_extractor, mask_generator, classifier, size, T):
        super(Network0, self).__init__()
        self._feature_extractor = feature_extractor
        self._mask_generator = mask_generator
        self._classifier = classifier
        self._gaussian_mask = GaussianMask(*size)
        self._T = T

    def forward(data):
        N = data.size()[0]
        mask = (th.rand(N, 2) - 0.5) / 2
        if data.is_cuda:
            mask = mask.cuda()
        prediction_list = []
        for _ in range(self._T):
            masked = mask * data
            features = self._feature_extractor(masked)

            mx, my, sx, sy = self._mask_generator(features)
            mask = self._gaussian_mask(mx, my, sx, sy)

            category = self._classifier(features)
            prediction_list.append(category)

        return prediction_list

    @staticmethod
    def loss(prediction_list, label):
        loss_list = []
        for prediction in prediction_list:
            ce = F.cross_entropy(prediction, label)
            loss_list.append(ce)

        return loss_list


class Network1(Network0):
    def __init__(self, feature_extractor, mask_generator, classifier, size, T,
                 gamma):
        super(Network1, self).__init__(feature_extractor, mask_generator,
                                       classifier, size, T)
        self._gamma = gamma(T)

    def loss(self, prediction_list, label):
        loss_list = []
        for prediction in prediction_list:
            ce = F.cross_entropy(prediction, label)
            gamma = self._gamma(i)
            loss_list.append(ce * gamma)

        return loss_list
