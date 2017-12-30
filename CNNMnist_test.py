import tensorflow as tf

# 加载MNIST数据
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 运行TensorFlow的InteractiveSession
# 通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的
sess = tf.InteractiveSession()

# 构建Softmax 回归模型
# 占位符
# 我们通过为输入图像和目标输出类别创建节点，来开始构建计算图。
# 输出图像（这里的x,y代表的不是具体值，而是一个占位符，可以在进行某一计算时，根据该占位符输入具体的值
# 输入图片x是一个二维的浮点数张量,这里分配给他的shape中，784是一张展平的MNIST图片的纬度（即28*28）.
# None表示大小不定，这里作为第一个纬度，用以代指batch（一批）的大小，即x的数量不定
x = tf.placeholder("float", shape=[None, 784])
# 输出类别值y_也是一个二维的张量，其中每一行为一个10纬的one-hot向量，用于代表对应的MNIST图片的类别
# 10纬的one-hot向量指[0,0,0,0,0,0,0,0,0]，举例：当一张MNIST图片上显示的数字为1时，那么对应的向量为[0,1,0,0,0,0,0,0,0]
y_ = tf.placeholder("float", shape=[None, 10])


# 变量


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化

# 卷积使用步长为1，边距为0，保证输入和输出的大小保持一致
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


# 池化使用2*2做模板
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 第一层卷积
# 在每个5x5的范围中算出32个特种值。卷积的权重张量是[5,5,1,32],前两个参数表示大小，第三个参数表示输入通道数，和输出通道数
W_conv1 = weight_variable([5, 5, 1, 32])
# 每个输出通道对应一个偏置量
b_conv1 = bias_variable([32])

# 这层我们把x变成一个4d的向量，2,3纬表示对应图片的宽高，最后一纬表示图片的颜色通道，灰度用1，rgb彩色用3
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 将x-image和权重向量进行卷积，再加上偏置项，然后应用ReLU激活函数，最后进行max_pooling.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
# 为了构建一个更深层的网络，我们将几个类似的层堆叠起来。第二层中，每5x5的patch中会得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
# 每个输出通道对应一个偏置量
b_conv2 = bias_variable([64])

# 将x-image和权重向量进行卷积，再加上偏置项，然后应用ReLU激活函数，最后进行max_pooling.
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
# 现在，图片减小的7x7，我们加入一个有1024个神经元的全连接层，来处理整个图片。
# 我们把池化层的输出张量reshape成一些向量，乘上权重矩阵，加上偏置量，再进行ReLU

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# 我们在输出层之前加入dropout为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
# 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，
# 还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
# 最后添加一个softmax层，就像前面单层的softmax regression一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
# 交叉熵，可以很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。
# 可以很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 获得训练的步骤
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 正确预测值
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 将预测值转换成float，并求出平均准确率
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
# 开始会话
sess.run(tf.global_variables_initializer())
# 训练两万次，每次取50数据量为一批，每100次迭代输出一次日志
for i in range(5000):
    print('i = %d ' % i)
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
