import tensorflow as tf
import mnist_data_download
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder("float", [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(8000):
    batch_xs, batch_ys = mnist_data_download.mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print(tf.argmax(y, 1))
print(tf.argmax(y_, 1))
correct_percent = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_percent, 'float'))

print(
    sess.run(accuracy, feed_dict={x: mnist_data_download.mnist.test.images, y_: mnist_data_download.mnist.test.labels}))
