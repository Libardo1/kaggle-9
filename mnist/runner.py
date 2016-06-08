import mnist.logistic_tf as logistic_tf
import mnist.kaggle      as kaggle

def go():
    labels = logistic_tf.train_and_test()
    kd     = kaggle.kaggle_data(labels)

    return kd

kaggle_data = go()
kaggle_data.to_csv(kaggle.DEFAULT_CSV_PATH, index=False, float_format='%.f')