import pandas as pd

DEFAULT_CSV_PATH = './submissions/kaggle.csv'

def kaggle_data(mnist_data):
    """
    A Kaggle submission for MNIST data is a csv file of the form
    -------------
    ImageId,Label
    0,a0,
    1,a1,
    ...
    i,ai
    ...
    n,an
    -------------
    where i is the image id in ai is an integer between 0 and 9 classifying 
    the digit.
    """
    image_ids = mnist_data.index
    labels    = mnist_data['label']
    cols      = ['ImageId', 'Label']
    kaggle_frame = pd.DataFrame({'ImageId': image_ids, 'Label': labels})

    return kaggle_frame
