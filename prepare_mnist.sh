if [ ! -f 'mnist.dat' ]
then
    if [ ! -f 'mnist.pkl.gz' ]
    then
        wget http://deeplearning.net/data/mnist/mnist.pkl.gz
    fi
    python prepare_mnist.py
fi
