from pdb import set_trace as st
from argparse import ArgumentParser
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from visdom import Visdom
from networks import MLP, Network0
from utilities import *
from visualizer import TraceVisualizer

parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mnist-path', type=str, default='mnist.dat')
parser.add_argument('--n-epochs', type=int, default=100)
parser.add_argument('--n-features', type=int, default=64)
parser.add_argument('--nonlinear', action=parse_nonlinear, default=F.relu)
parser.add_argument('--T', type=int, default=4)
parser.add_argument('--units', action=partition('-', int), default=(256,))
args = parser.parse_args()

if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)

loader_dict, size = create_mnist_loaders(args.mnist_path, args.batch_size)

n_pixels = size[0] * size[1]
feature_extractor = MLP(n_pixels, args.n_features, args.units, args.nonlinear)
model = Network0(feature_extractor, args.n_features, size, args.T)
if cuda:
    model.cuda()
optimizer = th.optim.Adam(model.parameters(), args.lr)

visdom = Visdom(env=__file__)
tl_vis = TraceVisualizer(visdom, {'title': 'training loss'})
ta_vis = TraceVisualizer(visdom, {'title': 'training accuracy'})
va_vis = TraceVisualizer(visdom, {'title': 'validation accuracy'})

for epoch in range(args.n_epochs):
    print 'epoch %d' % epoch

    tllist_list, talist_list = [], []
    for iteration, batch in enumerate(loader_dict['train']):
        data, labels = batch
        if cuda:
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)
        prediction_list = model(data)
        loss_list = cross_entropy(prediction_list, labels)
        tllist_list.append(loss_list)

        loss = sum(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy_list = accuracy(prediction_list, labels)
        talist_list.append(accuracy_list)

    for i, tllist in enumerate(zip(*tllist_list)):
        label = 'iteration %d' % i
        tltuple = tuple(l.data[0] for l in tllist)
        tl_vis.extend(tltuple, label)

    for i, talist in enumerate(zip(*talist_list)):
        label = 'iteration %d' % i
        ta_vis.extend(talist, label)

    valist_list = []
    for (data, labels) in loader_dict['validate']:
        if cuda:
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)
        prediction_list = model(data)
        accuracy_list = accuracy(prediction_list, labels)
        valist_list.append(accuracy_list)

    for i, valist in enumerate(zip(*valist_list)):
        va = sum(valist) / len(valist)
        label = 'iteration %d' % i
        va_vis.extend((va,), label)
