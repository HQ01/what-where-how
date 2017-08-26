from __future__ import division
import torch as th

class GaussianMask(nn.Module):
    def __init__(self, width, height):
        super(GaussianMask, self).__init__()
        self._width, self._height = width, height
        self._x = th.linspace(-1, 1, 2 / width)
        self._y = th.linspace(-1, 1, 2 / height)

    def forward(self, mu_x, mu_y, sigma_x, sigma_y):
        th.exp(-(self._x - mu_x) ** 2 / (2 * sigma_x ** 2))
