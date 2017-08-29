import cPickle as pickle
import gzip
import joblib

data_tuple = pickle.load(gzip.open('mnist.pkl.gz'))
data_list = []
for (data, labels) in data_tuple:
    data = (data - data.mean()) / data.std()
    data_list.append((data, labels))

joblib.dump(((28, 28), tuple(data_list)), 'mnist.dat')
