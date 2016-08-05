import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


FILETRAIN = './data/training.csv'
FILETEST = './data/test.csv'
FILELOOKUP = './data/IdLookupTable.csv'

def load(isTrain=True, cols=None):
    """
    Loads data for training or test indicated by isTrain
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FILETRAIN if isTrain else FILETEST
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if isTrain:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def load2d(isTrain=False, cols=None):
    """
    # make the matrix conpatible to the convolutional layer,
    # In this case, we use only the gray level image, so the color channel is 1.
    # -1 indicates it is compatible to the input dimension.
    """
    X, y = load(isTrain=isTrain, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y



if __name__ == '__main__':
    X, y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))

