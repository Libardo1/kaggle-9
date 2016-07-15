import mnist.logistic_tf as logistic_tf
import mnist.conv_tf     as conv_tf
import mnist.kaggle      as kaggle


def execute_analysis():
    labels = logistic_tf.train_and_test()
    kd     = kaggle.kaggle_data(labels)

    return kd


def execute_analysis_conv():
    labels = conv_tf.execute_analysis()
    kd     = kaggle.kaggle_data(labels)

    return kd

def main():
    kaggle_data = execute_analysis_conv()
    kaggle_data.to_csv(kaggle.DEFAULT_CSV_PATH, index=False, float_format='%.f')
