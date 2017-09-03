from __future__ import division
from pdb import set_trace as st
import time
import numpy as np
import tensorflow as tf
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utilities import cross_entropy


class CoreNetwork(nn.Module):
    def __init__(self):
        super(CoreNetwork, self).__init__()
        self._h_linear = nn.Linear(256, 256)
        self._g_linear = nn.Linear(256, 256)

    def forward(self, h, g):
        h = self._h_linear(h)
        g = self._g_linear(g)
        h = F.relu(h + g)
        return h


class GlimpseNetwork(nn.Module):
    def __init__(self, args):
        super(GlimpseNetwork, self).__init__()
        self._retina_size = args.n_scales * args.w * args.h
        self._retina_linear = nn.Linear(self._retina_size, 128)
        self._location_linear = nn.Linear(2, 128)
        self._hg_linear = nn.Linear(128, 256)
        self._hl_linear = nn.Linear(128, 256)

    def forward(self, retina, location):
        retina = retina.view(-1, self._retina_size)
        hg = F.relu(self._retina_linear(retina))
        hg = self._hg_linear(hg)

        hl = F.relu(self._location_linear(location))
        hl = self._hl_linear(hl)

        g = F.relu(hg + hl)
        return g


class LocationNetwork(nn.Module):
    def __init__(self, args):
        super(LocationNetwork, self).__init__()
        self._linear = nn.Linear(256, 2)
        self._sx, self._sy = args.sx, args.sy

    def forward(self, h):
        mean = self._linear(h)
        mx, my = th.chunk(mean, 2, 1)
        sx = th.ones(mx.size()) * self._sx
        sy = th.ones(my.size()) * self._sy
        if mean.is_cuda:
            sx, sy = sx.cuda(), sy.cuda()
        sx, sy = Variable(sx), Variable(sy)
        x, y = th.normal(mx, sx), th.normal(my, sy)
        x, y = x.detach(), y.detach()
        location = th.cat((x, y), 1)
        cache = ((mx, my, sx, sy), (x, y))
        return location, cache

    def loss(self, reward, cache):
        """
        Parameters
        ----------
        reward : N by 1 tensor
        cache : tuple
        """

        if cache is None:
            return 0

        (mx, my, sx, sy), (x, y) = cache
        log_px = -(x - mx) ** 2 / (2 * sx ** 2)
        log_py = -(y - my) ** 2 / (2 * sy ** 2)
        log_p = log_px + log_py
        value = th.mean(log_p * reward)
        return value


class RetinaEncoder(nn.Module):
    def __init__(self, args):
        super(RetinaEncoder, self).__init__()
        self._patch_size = np.array([args.w, args.h])
        self._n_scales = args.n_scales

        with tf.device('/cpu:0'):
            input = tf.placeholder("float32", None, 'input')
            glimpse_size = tf.placeholder("int32", None, 'glimpse_size')
            patch_size = tf.Variable(self._patch_size, False)
            offsets = tf.placeholder("float32", None, 'offsets')
            glimpse = tf.image.extract_glimpse(input, glimpse_size, offsets)
            self._glimpse = tf.image.resize_images(glimpse, self._patch_size)

        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

    def forward(self, input, offsets):
        cuda = input.is_cuda
        input, offsets = input.data.cpu().numpy(), offsets.data.cpu().numpy()
        input = np.transpose(input, (0, 3, 2, 1))
        feed_dict = {
            'input:0': input,
            'offsets:0': offsets,
            'glimpse_size:0': self._patch_size
        }
        glimpse_list = []
        for _ in range(self._n_scales):
            glimpse = self._session.run(self._glimpse, feed_dict)
            glimpse = np.transpose(glimpse, (0, 3, 2, 1))
            glimpse = th.from_numpy(glimpse)
            if cuda:
                glimpse = glimpse.cuda()
            glimpse_list.append(glimpse)
            feed_dict['glimpse_size:0'] *= 2

        retina = th.cat(glimpse_list, 1)
        retina = Variable(retina)
        return retina


class RAM(nn.Module):
    def __init__(self, args):
        super(RAM, self).__init__()
        self._classifier = nn.Linear(256, 10)
        self._core_network = CoreNetwork()
        self._glimpse_network = GlimpseNetwork(args)
        self._location_network = LocationNetwork(args)
        self._retina_encoder = RetinaEncoder(args)

        self._T = args.T

    def forward(self, data):
        N = data.size()[0]
        location, cache = (th.rand(N, 2) - 0.5) * 2, None
        h = th.zeros(N, 256)
        if data.is_cuda:
            location, h = location.cuda(), h.cuda()
        location, h = Variable(location), Variable(h)
        prediction_list, cache_list = [], []
        for _ in range(self._T):
            retina = self._retina_encoder(data, location)
            glimpse = self._glimpse_network(retina, location)
            h = self._core_network(h, glimpse)
            category = self._classifier(h)
            prediction_list.append(category)
            cache_list.append(cache)
            location, cache = self._location_network(h)

        return prediction_list, cache_list

    def loss(self, prediction_list, cache_list, label):
        ce_tuple = tuple(cross_entropy(prediction_list, label))
        ce_loss = sum(ce_tuple)

        argmax = lambda t: th.max(t, 1)[1]
        category_tuple = tuple(argmax(p) for p in prediction_list)
        indicator_tuple = tuple(c == label for c in category_tuple)
        reward, rl_loss = 0, 0
        for indicator, cache in reversed(zip(indicator_tuple, cache_list)):
            reward = reward + indicator
            rl_loss = rl_loss + self._location_network.loss(reward.float(), cache)

        value = (ce_loss + rl_loss) / self._T
        return value, ce_tuple
