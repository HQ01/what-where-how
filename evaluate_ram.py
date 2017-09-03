from pdb import set_trace as st
from argparse import ArgumentParser
from random import randint
import numpy as np
import torch as th
from torch.autograd import Variable
from visdom import Visdom
from ram import RAM
from utilities import *
from visualizer import TraceVisualizer, ImageVisualizer

parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--gamma_sx', type=float, default=0.99)
parser.add_argument('--gamma_sy', type=float, default=0.99)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--h', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mnist-path', type=str, default='mnist.dat')
parser.add_argument('--n-epochs', type=int, default=100)
parser.add_argument('--n-scales', type=int, default=1)
parser.add_argument('--sx', type=float, default=1)
parser.add_argument('--sy', type=float, default=1)
parser.add_argument('--T', type=int, default=4)
parser.add_argument('--vis-glimpse', action='store_true')
parser.add_argument('--vis-height', type=int, default=100)
parser.add_argument('--vis-interval', type=int, default=10)
parser.add_argument('--vis-location', action='store_true')
parser.add_argument('--vis-width', type=int, default=100)
parser.add_argument('--w', type=int, default=8)
args = parser.parse_args()
print args

if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)

loader_dict, size = create_mnist_loaders(args.mnist_path, args.batch_size)

model = RAM(args)
if cuda:
    model.cuda()
optimizer = th.optim.Adam(model.parameters(), args.lr)

visdom = Visdom(env=__file__)
tl_vis = TraceVisualizer(visdom, {'title': 'training loss'})
ta_vis = TraceVisualizer(visdom, {'title': 'training accuracy'})
va_vis = TraceVisualizer(visdom, {'title': 'validation accuracy'})


class GlimpseVisualizer(object):
    def __init__(self, T, n_scales, opts):
        super(GlimpseVisualizer, self).__init__()
        self._v_tuple_tuple = tuple(
            tuple(ImageVisualizer(visdom, opts) for _ in range(n_scales))
            for t in range(T))

    def visualize(self, internal):
        data = internal['data'].data.cpu().numpy()
        N = data.shape[0]
        index = randint(0, N - 1)
        glimpse_list = internal['glimpse_list']
        glimpse_tuple = tuple(g.data[index].cpu().numpy()
                              for g in glimpse_list)
        for glimpse, v_tuple in zip(glimpse_tuple, self._v_tuple_tuple):
            g_tuple = np.split(glimpse, glimpse.shape[0], 0)
            for g, v in zip(g_tuple, v_tuple):
                g = np.reshape(g, (args.w, args.h))
                v.visualize(g)


class LocationVisualizer(object):
    def __init__(self, w, h, pw, ph, T, opts):
        super(LocationVisualizer, self).__init__()
        self._w, self._h = w, h
        self._pw, self._ph = pw, ph
        self._v_tuple = tuple(ImageVisualizer(visdom, opts) for _ in range(T))

    def _mask(self, location):
        """
        Parameters
        ----------
        location : numpy.ndarray ranging from -1 to 1 (N * 2)
        """

        location = (location + 1) / 2
        x, y = np.split(location, 2, 0)
        x *= self._w
        y *= self._h
        x, y = x.astype(np.int), y.astype(np.int)
        x_indices = tuple(np.arange(self._w) for _ in range(self._h))
        x_indices = tuple(np.reshape(x, (1, self._w)) for x in x_indices)
        x_indices = np.concatenate(x_indices, 0)
        y_indices = tuple(np.arange(self._h) for _ in range(self._w))
        y_indices = tuple(np.reshape(y, (self._h, 1)) for y in y_indices)
        y_indices = np.concatenate(y_indices, 1)
        indicator = np.logical_and((np.abs(x_indices - x) < self._pw),
                                   (np.abs(y_indices - y) < self._ph))
        mask = np.zeros((self._w, self._h))
        mask[indicator] = 1
        return mask

    def visualize(self, internal):
        data = internal['data'].data.cpu().numpy()
        N = data.shape[0]
        index = randint(0, N - 1)
        data = np.squeeze(data[index])
        location_list = internal['location_list']
        location_tuple = tuple(l.data[index].cpu().numpy()
                               for l in location_list)
        for l, v in zip(location_tuple, self._v_tuple):
            mask = self._mask(l)
            v.visualize(mask * data)


opts = {'width': args.vis_width, 'height': args.vis_height}
g_vis = GlimpseVisualizer(args.T, args.n_scales, opts)
w, h = size
l_vis = LocationVisualizer(w, h, args.w, args.h, args.T, opts)

sx, sy = args.sx, args.sy
for epoch in range(args.n_epochs):
    print 'epoch %d' % epoch

    model.train()
    model.configure(sx=sx, sy=sy)
    tllist_list, talist_list = [], []
    for iteration, batch in enumerate(loader_dict['train']):
        data, labels = batch
        data = data.view(args.batch_size, 1, *size)
        if cuda:
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)
        prediction_list, internal, cache_list = model(data)
        loss, ce_list = model.loss(prediction_list, cache_list, labels)
        tllist_list.append(ce_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy_list = accuracy(prediction_list, labels)
        talist_list.append(accuracy_list)
        if (iteration + 1) % args.vis_interval == 0:
            if args.vis_glimpse:
                g_vis.visualize(internal)
            if args.vis_location:
                l_vis.visualize(internal)

    sx *= args.gamma_sx
    sy *= args.gamma_sy

    for i, tllist in enumerate(zip(*tllist_list)):
        label = 'iteration %d' % i
        tltuple = tuple(l.data[0] for l in tllist)
        tl_vis.extend(tltuple, label)

    for i, talist in enumerate(zip(*talist_list)):
        label = 'iteration %d' % i
        ta_vis.extend(talist, label)

    model.eval()
    valist_list = []
    for (data, labels) in loader_dict['validate']:
        data = data.view(args.batch_size, 1, *size)
        if cuda:
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)
        prediction_list, internal = model(data)
        accuracy_list = accuracy(prediction_list, labels)
        valist_list.append(accuracy_list)

    for i, valist in enumerate(zip(*valist_list)):
        va = sum(valist) / len(valist)
        label = 'iteration %d' % i
        va_vis.extend((va, ), label)
