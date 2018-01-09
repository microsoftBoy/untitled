import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 1), dtype=np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(filter_data))
    # print(sess.run(tf.shape(input_data)))
    print('======================')
    # print(sess.run(y))
    print('----------------------')
    print(sess.run(tf.shape(y)))
