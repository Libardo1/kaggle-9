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

x = tf.placeholder(tf.float32, shape=[None, INPUT_DIMENSIONS])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIMENSIONS])

# Weight matrix and bias vector.
W = tf.Variable(tf.zeros([INPUT_DIMENSIONS, OUTPUT_DIMENSIONS]))
b = tf.Variable(tf.zeros([OUTPUT_DIMENSIONS]))

sess.run(tf.initialize_all_variables())

# Estimator for the outputs
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Bayes classifiers
estimated_classifier = lambda y: tf.argmax(y,1)
training_classifier  = tf.argmax(y_,1)

# Risk estimators
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

# Train the model
mnist_train = mnist.fetch_data(mnist.TRAIN)

for batch in mnist.batches(mnist_train, batch_size=BATCH_SIZE):
    labels   = batch['label']
    features = batch.T.iloc[1:].T
    train_step.run(feed_dict={x: features, y_: labels}, session=sess)


correct_prediction = tf.equal(estimated_classifier(y), training_classifier)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test the trained model
mnist_test = mnist.fetch_data(mnist.TEST)
images = mnist_test.T.iloc[1:].T
labels = mnist_test['label']

print(accuracy.eval(feed_dict={x: images, y_: labels}))

sess.close()