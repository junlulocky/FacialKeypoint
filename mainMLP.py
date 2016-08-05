"""
MLP Structure:

  #  name      size
---  ------  ------
  0  input     9216
  1  hidden     100
  2  output      30

"""

# user defined libraries
from loader import *

# third party libraries
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne.base import BatchIterator

from matplotlib import pyplot

from utils import plot_sample


net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch


    hidden_num_units=100,  # number of units in hidden layer
    hidden_nonlinearity=rectify,  # the default one is rectify function

    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    train_split=TrainSplit(eval_size=0.25),
    objective_l2=0.0025,  # l2 regularizer
    objective_loss_function = squared_error,
    batch_iterator_train=BatchIterator(batch_size=200),
    batch_iterator_test=BatchIterator(batch_size=200),
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )


def test():
    # learnin curve
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()

    # plot samples
    X, _ = load(isTrain=False)
    y_pred = net1.predict(X)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    pyplot.show()






if __name__ == '__main__':
    X, y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))

    net1.fit(X, y)
    test()
    plot_sample()
