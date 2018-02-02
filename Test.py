from __future__ import print_function

# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# node1 = tf.constant(3.0, dtype=tf.float32)
# node2 = tf.constant(4.0)  # also tf.float32 implicitly
# print(node1, node2)
#
# sess = tf.Session()
# print(sess.run([node1, node2]))
#
#
# node3 = tf.add(node1, node2)
# print("node3:", node3)
# print("sess.run(node3):", sess.run(node3))


# import os
# dirs = dir(os)
# print(dirs)
# for i in dirs:
#     print(i)

# print(dir())
# a = 100
#
#
# def aa():
#     b = 100
#
# dirs = dir()
# dirs.append('123')
# print(dir())
# print(dirs)
# print(len(dir()))

# shoplist = ['apple', 'mango', 'carrot', 'banana']
# mylist = shoplist
# print(mylist)
#
# del shoplist[0]
# print(mylist)


import time
import os

source = ['G:\MY-JAVA']
target_dir = 'G:\\MY-JAVA\\Backup\\'

target = target_dir + time.strftime('%Y%m%d%H%M%S') + '.zip'
print(target)
# zip_comand = "zip-qr'%s'%s" % (target, ''.join(source))
# if os.system(zip_comand) == 0:
#     print('Successful backup to ',target)
# else:
#     print('Backup Failed')

exit = os.path.exists(''.join(target_dir))
print(exit)