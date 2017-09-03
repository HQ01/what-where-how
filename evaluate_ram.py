from argparse import ArgumentParser
import torch as th
from torch.autograd import Variable
from visdom import Visdom
from ram import RAM
from utilities import *
from visualizer import TraceVisualizer

parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--gamma_sx', type=float, default=0.9)
parser.add_argument('--gamma_sy', type=float, default=0.9)
parser.add_argument('--h', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mnist-path', type=str, default='mnist.dat')
parser.add_argument('--n-epochs', type=int, default=100)
parser.add_argument('--n-scales', type=int, default=1)
parser.add_argument('--sx', type=float, default=1)
parser.add_argument('--sy', type=float, default=1)
parser.add_argument('--T', type=int, default=4)
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
        prediction_list, cache_list = model(data)
        loss, ce_list = model.loss(prediction_list, cache_list, labels)
        tllist_list.append(ce_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy_list = accuracy(prediction_list, labels)
        talist_list.append(accuracy_list)

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
        prediction_list = model(data)
        accuracy_list = accuracy(prediction_list, labels)
        valist_list.append(accuracy_list)

    for i, valist in enumerate(zip(*valist_list)):
        va = sum(valist) / len(valist)
        label = 'iteration %d' % i
        va_vis.extend((va, ), label)
