import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print('number of train is %d'%(mnist.train.num_examples))
print('number of test is %d'%(mnist.test.num_examples))
print('number of validation is %d'%(mnist.validation.num_examples))

