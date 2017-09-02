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
parser.add_argument('--nonlinear', action=parse_nonlinear, default=F.relu)
parser.add_argument('--T', type=int, default=20)
parser.add_argument('--units', action=partition('-', int), default=(256, ))
args = parser.parse_args()

if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)

loader_dict, size = create_mnist_loaders(args.mnist_path, args.batch_size)
n_pixels = size[0] * size[1]


def model(data, labels):
    network = MLP(n_pixels, 10, args.units, args.nonlinear)
    if cuda:
        network.cuda()
    optimizer = th.optim.Adam(network.parameters(), args.lr)
    prediction_list = []
    for i in range(args.T):
        category = network(data)
        prediction_list.append(category)

        if i != args.T - 1:
            loss = F.cross_entropy(category, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return prediction_list


visdom = Visdom(env=__file__)
va_vis = TraceVisualizer(visdom, {'title': 'validation accuracy'})

valist_list = []
for (data, labels) in loader_dict['validate']:
    if cuda:
        data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    prediction_list = model(data, labels)
    accuracy_list = accuracy(prediction_list, labels)
    valist_list.append(accuracy_list)

zipped = zip(*valist_list)
for i, valist in enumerate(zipped):
    label = 'iteration %d' % i
    va_vis.extend(valist, label)

va = sum(zipped[-1]) / len(zipped[-1])
print va
