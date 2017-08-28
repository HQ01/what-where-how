from __future__ import division
import math
import torch as th


class GaussianMask(nn.Module):
    def __init__(self, width, height):
        super(GaussianMask, self).__init__()
        self._width, self._height = width, height
        self._x = th.linspace(-1, 1, 2 / width)
        self._y = th.linspace(-1, 1, 2 / height)

    def forward(self, mx, my, sx, sy):
        x, y = self._x, self._y
        z = ((x - mx) / sx)**2 + ((y - my) / sy)**2
        mask = 1 / (2 * math.pi * sx * sy) * th.exp(-z / 2)
        return mask
