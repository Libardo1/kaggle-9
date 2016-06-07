import pandas as pd


TRAIN = 'data/mnist/train.csv'
TEST  = 'data/mnist/test.csv'

def uncompress_mnist_data(path):
    pass

def fetch_data(handle):
    return pd.read_csv(handle)

def mnist_feature_vectors_and_labels(data):
    labels = data['label']
    images = data.T.iloc[1:].T

    return labels, images

def output_classifications():
    pass

def batches(df, batch_size=1):
    def update():
        batch_begin += batch_size
        batch_end   += batch_size
        
    batch_begin = 0
    batch_end   = batch_begin + batch_size
    
    while batch_begin <= df.shape[0]:
        yield df.iloc[batch_begin:batch_end]
        
        update()
