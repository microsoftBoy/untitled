import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
output = tf.nn.max_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
