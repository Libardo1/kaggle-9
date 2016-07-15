"""
Convnet classifier for MNIST data using TensorFlow.
"""
import tensorflow  as tf
import mnist.mnist as mnist
import pandas      as pd


DEFAULT_INPUT_DIMENSIONS  = 784
DEFAULT_OUTPUT_DIMENSIONS = 10
DEFAULT_LEARNING_RATE     = 0.1
DEFAULT_BATCH_SIZE        = 50
DEFAULT_KEEP_PROB         = 0.5


class CNNClassifier():
    def __init__(self, 
                 x,
                 y_,
                 y_conv,
                 objective_func, 
                 train_step,
                 predict,
                 keep_prob
                 ):

        self.x      = x
        self.y_     = y_
        self.y_conv = y_conv
        self.objective_func = objective_func
        self.train_step     = train_step
        self.predict        = predict
        self.keep_prob      = keep_prob


    def init(self, session):
        """
        Initialize the conv net.
        """
        session.run(tf.initialize_all_variables())


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')


def conv_hlayer1(x):
    """
    First hidden layer of convolutional neural net.
    """
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #Output of hidden layer.
    h_pool1 = max_pool_2x2(h_conv1)

    return h_pool1


def conv_hlayer2(h_pool1):
    """
    Second hidden layer of convolutional neural net.
    """
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    return h_pool2


def conv_hlayer3(h_pool2):
    """
    Third hidden layer of convolutional neural net with dropout.
    """
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1        = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob  = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    return h_fc1_drop, keep_prob


def conv_output_layer(h_fc1_drop):
    """
    Readout layer
    """
    W_fc2  = weight_variable([1024, 10])
    b_fc2  = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv


def build_nn(x):
    """
    Construct the convolutional neural network and return the output layer variable.
    """
    h_pool1    = conv_hlayer1(x)
    h_pool2    = conv_hlayer2(h_pool1)
    h_fc1_drop, keep_prob = conv_hlayer3(h_pool2)
    y_conv     = conv_output_layer(h_fc1_drop)

    return y_conv, keep_prob


def build_computational_graph(input_dimensions=DEFAULT_INPUT_DIMENSIONS, 
                              output_dimensions=DEFAULT_OUTPUT_DIMENSIONS):
    """
    Construct the computational graph for training and operating the 
    convolutional neural network. Everything is bundled together into an
    object so that it can be conveniently worked with for training and testing.
    """

    x  = tf.placeholder(tf.float32, shape=[None, input_dimensions])
    y_ = tf.placeholder(tf.float32, shape=[None, output_dimensions])
    
    # Convnet to be used for classifying MNIST digits.
    y_conv, keep_prob = build_nn(x)

    # Objective function.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    # Training step.
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    predict = tf.argmax(y_conv, 1)

    conv_net = CNNClassifier(x, y_, y_conv, cross_entropy, train_step, predict, keep_prob)

    return conv_net


def train_model(conv_net,
                session, 
                mnist_train, 
                batch_size=DEFAULT_BATCH_SIZE,
                keep_prob=DEFAULT_KEEP_PROB,
                logging=False):
    """
    Train the convolutional neural net.
    """
    for (labels, images) in mnist.batches(mnist_train, batch_size=batch_size):
        feed_dict = {
                        conv_net.x: images, 
                        conv_net.y_: labels, 
                        conv_net.keep_prob: keep_prob
                    }
        conv_net.train_step.run(feed_dict=feed_dict, session=session)


def test_eval(conv_net, mnist_test):
    feed_dict = {
            conv_net.x : mnist_test.images,
            conv_net.y_ : mnist_test.labels,
            conv_net.keep_prob: 1.0
        }
    accuracy_eval = conv_net.accuracy_metric.eval(feed_dict=feed_dict)

    return accuracy_eval


def test_model(conv_net, mnist_test, session):
    """
    Classifiy the test data.
    """
    idxs = ['label']
    label_frame = pd.DataFrame(None, None, idxs)
    image_frame = mnist_test.images

    for item in mnist.unlabeled_batches(mnist_test):
        idx       = item.index[0]
        feed_dict = {
                conv_net.x: item, 
                conv_net.keep_prob: 1.0
            }
        label     = session.run(conv_net.predict, feed_dict=feed_dict)
        label_frame.loc[idx] = label

    # Join the frames together and return them.
    output_frame = label_frame.join(image_frame)
    output_frame.label.astype(int)

    return output_frame


def execute_analysis(train=mnist.TRAIN, test=mnist.TEST):
    # Start sessions and initialize conv net.
    sess = tf.Session()
    conv_net = build_computational_graph()
    conv_net.init(sess)
    
    # Train the model
    mnist_train = mnist.from_data_frame(mnist.fetch_data(train))
    train_model(conv_net, sess, mnist_train)
    
    # Test the model.
    mnist_test = mnist.from_unlabeled_data_frame(mnist.fetch_data(test))
    labels = test_model(conv_net, mnist_test, sess)

    sess.close()

    return labels
