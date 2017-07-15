import tensorflow as tf 
import cPickle as pickle
import numpy as np 
# a = tf.Variable(np.ones((3,33,10)))
# b = tf.expand_dims(tf.Variable(np.arange(33) + 0.0),-1)
# print b
# c = tf.transpose(a,perm = [1,0]) * b
# c = tf.multiply(a,b)
# d = tf.ones([10,2])
a = [23.12,34.23,12.56]
b = tf.nn.l2_normalize(a,0)
# initializer = (np.array(0), np.array(1))
# fibonaccis = tf.scan(lambda a, _: (a[1], a[0] + a[1]), elems)
with tf.Session() as sess:

	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
	# print sess.run(a)
	print sess.run(b)
	# print sess.run(c)
	# print sess.run(d)

# import numpy as np
# import matplotlib.pyplot as plt
# # alpha = ['ABC', 'DEF', 'GHI', 'JKL']
# d = pickle.load(open('attention.file'))
# print d[1][0]
# exit()
# # print len(d)
# data = d[0]
# print data
# # print d[0][0]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(data, cmap = plt.cm.Blues)
# fig.colorbar(cax)

# # ax.set_xticklabels(['']+alpha)
# # ax.set_yticklabels(['']+alpha)

# plt.show()

# a = []

# b = np.ones((10,10))
# c = np.random.rand(10,20)
# print c[0]
# for b1,c1 in zip(b,c):
# 	a.extend((b1,c1))

# print a[1]
# import pandas as pd 
# file = 'data/nlpcc/train.txt'
# df = pd.read_csv(file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna('')
# df['alen'] = df.apply(lambda x:len(x['answer'].split()),axis = 1)
# print df[df['flag'] == 1]['alen'].
# a = ('a','b')
# print str(a)

