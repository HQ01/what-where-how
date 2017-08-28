import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, out_features, features, linear=nn.Linear, nonlinear=F.relu):
        super(MLP, self).__init__()
        shapes = zip((in_features,) + features, features + (out_features,))
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
