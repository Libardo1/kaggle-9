"""
Logistic classifier using TensorFlow for MNIST data.
"""
import tensorflow   as tf
import numpy        as np
import pandas       as pd
import mnist.mnist  as mnist
import mnist.kaggle as kaggle

"""
The MNIST data has 784 dimensions. Each vector represents a 28x28 pixel grayscale
image of a handwritten number between 0 and 9. The number of possible classifications is 10,
one for each digit in 0..9.
"""
INPUT_DIMENSIONS  = 784
OUTPUT_DIMENSIONS = 10
LEARNING_RATE     = 0.1
BATCH_SIZE        = 100

def build_computational_graph(input_dimensions=INPUT_DIMENSIONS, 
                              output_dimensions=OUTPUT_DIMENSIONS,
                              learning_rate=LEARNING_RATE,
                              batch_size=BATCH_SIZE):
    """
    Build the computational graph representing the statistical model that is
    going to be used to predict digit classifications.
    """
    x  = tf.placeholder(tf.float32, shape=[None, input_dimensions])
    y_ = tf.placeholder(tf.float32, shape=[None, output_dimensions])

    # Weight matrix and bias vector.
    W = tf.Variable(tf.zeros([input_dimensions, output_dimensions]))
    b = tf.Variable(tf.zeros([output_dimensions]))

    # Estimator for the outputs
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Bayes classifiers
    predict = tf.argmax(y,1)

    # Risk estimators
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y + 1e-10), reduction_indices=[1]))

    # TODO: try another optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # Return the model components
    return (x, y_, W, b, y, predict, cross_entropy, train_step)


def train_model(mnist_train, session, train_step, x, y_):
    for (labels, images) in mnist.batches(mnist_train, batch_size=BATCH_SIZE):
        train_step.run(feed_dict={x: images, y_: labels}, session=session)


def test_model(mnist_test, session, classifier, x):
    """
    Classifies test data. This is a test of how good the model fits the data.
    """
    idx = ['label']
    label_frame = pd.DataFrame(None, None, ['label'])
    image_frame = mnist_test.images

    for item in mnist.unlabeled_batches(mnist_test):
        idx   = item.index[0]
        label = session.run(classifier, feed_dict={x: item})
        label_frame.loc[idx] = label

    output_frame = label_frame.join(image_frame)
    output_frame.label.astype(int)

    return output_frame


def train_and_test():
    x, y_, W, b, y, predict, cross_entropy, train_step = build_computational_graph()

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    # Train the model
    mnist_train = mnist.from_data_frame(mnist.fetch_data(mnist.TRAIN))
    train_model(mnist_train, session, train_step, x, y_)

    # Run the model on the testing data.
    mnist_test = mnist.from_unlabeled_data_frame(mnist.fetch_data(mnist.TEST))
    labels = test_model(mnist_test, session, predict, x)
    
    session.close()

    return labels