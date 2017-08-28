from pdb import set_trace as st
import numpy as np


class Visualizer(object):
    def __init__(self, start):
        super(Visualizer, self).__init__()
        self._start = start

    def extend(self, s, clear=False):
        self._extend(s)
        self._start += len(s)
        if clear:
            del s[:]


class VisdomVisualizer(Visualizer):
    def __init__(self, visdom, options={}, start=0):
        super(VisdomVisualizer, self).__init__(start)
        self._visdom = visdom
        self._window = None
        self._options = options

    def _extend(self, s):
        X = np.arange(self._start, self._start + len(s))
        Y = np.array(s)
        if self._window:
            self._window = self._visdom.line(
                Y, X, self._window, opts=self._options, update='append')
        else:
            self._window = self._visdom.line(Y, X, opts=self._options)


class TraceVisualizer(VisdomVisualizer):
    def __init__(self, visdom, options={}, start=0):
        super(TraceVisualizer, self).__init__(visdom, options, start)
        self._window = self._visdom.line(np.zeros((1,)), opts=options) # TODO
        self._start_dict = {}

    def extend(self, s, label, clear=False):
        start = self._start_dict.setdefault(label, self._start)
        X = np.arange(self._start, self._start + len(s))
        Y = np.array(s)
        self._window = self._visdom.updateTrace(X, Y, self._window, name=label)
        self._start_dict[label] += len(s)
        if clear:
            del s[:]


class TensorboardVisualizer(Visualizer):
    _logger = __import__('tensorboard_logger')

    def __init__(self, name, start=0):
        super(TensorboardVisualizer, self).__init__(start)
        self._name = name

    @staticmethod
    def configure(path):
        TensorboardVisualizer._logger.configure(path)

    def _extend(self, s):
        for index, value in enumerate(s):
            self._logger.log_value(self._name, value, self._start + index)
