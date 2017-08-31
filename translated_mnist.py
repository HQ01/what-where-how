from argparse import ArgumentParser
from random import randint
import joblib
import numpy as np

parser = ArgumentParser()
parser.add_argument('--height', type=int, default=56)
parser.add_argument('--mnist-path', type=str, default='mnist.dat')
parser.add_argument('--width', type=int, default=56)
parser.add_argument('--x', type=int, default=-1)
parser.add_argument('--y', type=int, default=-1)
args = parser.parse_args()
print args

size, datasets = joblib.load(args.mnist_path)
dataset_list = []
for (data, labels) in datasets:
    data_list = []
    for d in data:
        canvas = np.zeros((args.width, args.height))
        if args.x < 0 or args.y < 0:
            x, y = randint(0, 
        else:
            x, y = args.x, args.y
