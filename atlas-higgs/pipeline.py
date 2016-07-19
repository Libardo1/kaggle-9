import pandas as pd
import numpy as np


class LabeledBatch():
    """
    LabeledBatch is an object for wrapping DataFrames for the ATLAS experiment. 
    This makes it more convenient to separate the data and labels for processing. 
    It also also enables efficiently iterating through the data set using (mini-)batches. 
    The goal of the data pipeline is to classify collisions events as signal 
    (Higgs -> tau + tau decays) or as background (other kinds of collision processes).
    """
    def __init__(self, data, labels):
        self._data   = data
        self._labels = labels
        self._length = data.shape[0]
        self._shape  = data.shape

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def length(self):
        return self._length

    @property
    def shape(self):
        return self._shape
    

class UnlabeledBatch():
    """
    UnlabeledBatch is an object for wrapping DataFrames for the ATLAS experiment testing data. 
    This makes it more convenient to separate the data and labels for processing. 
    It also also enables efficiently iterating through the data set using (mini-)batches. 
    The goal of the data pipeline is to classify collisions events as signal 
    (Higgs -> tau + tau decays) or as background (other kinds of collision processes).
    """
    def __init__(self, data):
        self._data   = data
        self._shape  = data.shape

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape
    

class BatchIterator():
    """
    BatchIterator is a generator for batch processing of the Kaggle ATLAS experiment data.
    """  
    def __init__(self, atlas_df, batch_size=1):
        self.df          = atlas_df
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
        data = self.df.data.iloc[self.batch_begin : self.batch_end]

        if self.batch_end < self.df.length and self.batch_begin < self.df.length:
            self.batch_begin += self.batch_size
            self.batch_end   += self.batch_size

            return labels, data

        elif self.batch_end >= self.df.length and self.batch_begin < self.df.length:
            # Shorten the last batch
            self.batch_begin += self.batch_size
            self.batch_end    = self.df.length

            return labels, data
        else:
            raise StopIteration


class UnlabeledBatchIterator():
    """
    Creates batches for classification.
    """    
    def __init__(self, atlas_df, batch_size=1):
        self.df          = atlas_df
        self.batch_begin = 0
        self.batch_end   = self.batch_begin + batch_size
        self.batch_size  = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        data = self.df.data.iloc[self.batch_begin : self.batch_end]

        if self.batch_end < self.df.length and self.batch_begin < self.df.length:
            self.batch_begin += self.batch_size
            self.batch_end   += self.batch_size

            return data

        elif self.batch_end >= self.df.length and self.batch_begin < self.df.length:
            # Shorten the last batch
            self.batch_begin += self.batch_size
            self.batch_end    = self.df.length

            return data
        else:
            raise StopIteration


def batches(atlas_df, batch_size=1):
    return BatchIterator(atlas_df, batch_size)


def unlabeled_batches(atlas_df, batch_size=1):
    return UnlabeledBatchIterator(atlas_df, batch_size)


def from_data_frame(df, label_col='Label'):
    try:
        labels = df[label_col]
        data   = df.T.iloc[1:].T
    except KeyError as e:
        raise e

    return (data, labels)


def from_unlabeled_data_frame(df):
    return UnlabeledBatch(df)


def decompress_dataset(path):
    """
    Uncompress a tarball file containing Atlas-Higgs data.
    """
    pass


def fetch_data(handle):
    return pd.read_csv(handle)
