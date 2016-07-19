import pandas as pd
import numpy  as np


TRAIN = 'data/mnist/train.csv'
TEST  = 'data/mnist/test.csv'

class Mnist():
    """
    Mnist is a class for wrapping up the data frames for Mnist data. This makes it more convenient to
    iterate over them using a generator or other iterator type for doing minibatch data processing.
    """
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._length = images.shape[0]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def length(self):
        return self._length


class MnistBatch():
    """
    Creates batches for doing regression over the labels.
    """    
    def __init__(self, mnist_df, batch_size=1):
        self.df          = mnist_df
        self.batch_begin = 0
        self.batch_end   = self.batch_begin + batch_size
        self.batch_size  = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _new_labels(self, labels, idx_range):
        idx  = pd.Index(np.arange(0,10))
        rows = labels.shape[0]
        cols = idx.shape[0]
        new_labels = pd.DataFrame(np.zeros((rows, cols)), np.arange(idx_range[0], idx_range[1]), idx)

        for i in new_labels.index:
            new_labels.loc[i, labels.loc[i]] = 1

        return new_labels

    def next(self):
        labels = self._new_labels(self.df.labels.iloc[self.batch_begin : self.batch_end],\
                                 (self.batch_begin, self.batch_end))
        images = self.df.images.iloc[self.batch_begin : self.batch_end]

        if self.batch_end < self.df.length and self.batch_begin < self.df.length:
            self.batch_begin += self.batch_size
            self.batch_end   += self.batch_size

            return labels, images

        elif self.batch_end >= self.df.length and self.batch_begin < self.df.length:
            # Shorten the last batch
            self.batch_begin += self.batch_size
            self.batch_end    = self.df.length

            return labels, images
        else:
            raise StopIteration


class UnlabeledMnistBatch():
    """
    Creates batches for doing regression over the labels.
    """    
    def __init__(self, mnist_df, batch_size=1):
        self.df          = mnist_df
        self.batch_begin = 0
        self.batch_end   = self.batch_begin + batch_size
        self.batch_size  = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        images = self.df.images.iloc[self.batch_begin : self.batch_end]

        if self.batch_end < self.df.length and self.batch_begin < self.df.length:
            self.batch_begin += self.batch_size
            self.batch_end   += self.batch_size

            return images

        elif self.batch_end >= self.df.length and self.batch_begin < self.df.length:
            # Shorten the last batch
            self.batch_begin += self.batch_size
            self.batch_end    = self.df.length

            return images
        else:
            raise StopIteration


def batches(mnist_df, batch_size=1):
    return MnistBatch(mnist_df, batch_size)


def unlabeled_batches(mnist_df, batch_size=1):
    return UnlabeledMnistBatch(mnist_df, batch_size)


def from_data_frame(df, label_col='label'):
    try:
        labels = df[label_col]
        images = df.T.iloc[1:].T
    except KeyError as e:
        raise e

    return Mnist(images, labels)


def from_unlabeled_data_frame(df):
    return Mnist(df, None)


def decompress_mnist_data(path):
    """
    Uncompress a tarball file containing mnist data.
    """
    pass


def fetch_data(handle):
    return pd.read_csv(handle)
