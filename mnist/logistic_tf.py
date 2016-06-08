"""
Logistic classifier using TensorFlow for MNIST data.
"""
import tensorflow  as tf
import numpy       as np
import pandas      as pd
import mnist.mnist as mnist 

"""
The MNIST data has 784 dimensions. Each vector represents a 28x28 pixel grayscale
image of a handwritten number between 0 and 9. The number of possible classifications is 10,
one for each digit in 0..9.
"""
INPUT_DIMENSIONS  = 784
OUTPUT_DIMENSIONS = 10
LEARNING_RATE     = 0.5
BATCH_SIZE        = 100

sess = tf.Session()

x  = tf.placeholder(tf.float32, shape=[None, INPUT_DIMENSIONS])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIMENSIONS])

# Weight matrix and bias vector.
W = tf.Variable(tf.zeros([INPUT_DIMENSIONS, OUTPUT_DIMENSIONS]))
b = tf.Variable(tf.zeros([OUTPUT_DIMENSIONS]))

sess.run(tf.initialize_all_variables())

# Estimator for the outputs
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Bayes classifiers
classifier = tf.argmax(y,1)

# Risk estimators
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

# Train the model
mnist_train = mnist.from_data_frame(mnist.fetch_data(mnist.TRAIN))

for (labels, images) in mnist.batches(mnist_train, batch_size=BATCH_SIZE):
    train_step.run(feed_dict={x: images, y_: labels}, session=sess)

# Run the model on the testing data.
mnist_test = mnist.from_unlabeled_data_frame(mnist.fetch_data(mnist.TEST))

for batch in mnist.unlabeled_batches(mnist_test, batch_size=BATCH_SIZE):
    sess.run(classifier, feed_dict={x: batch})


sess.close()