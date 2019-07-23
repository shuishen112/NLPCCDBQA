'''
@Description: this is the attentive pooling network of the question answering
@Author: zhansu
@Date: 2019-07-10 21:50:33
@LastEditTime: 2019-07-23 17:11:59
@LastEditors: Please set LastEditors
'''

import tensorflow as tf
from models.basis_model import Model


class Attentive_CNN(Model):

    def attentive_pooling(self, input_left, input_right):
        """
        docstring here: attentive pooling network
            :param self:
            :param input_left: question [batch,q_len,vector_size(num_filters * num_of_window)]
            :param input_right: answer [batch,a_len,vector_size(num_filters * num_of_window)]
        """

        self.q_len = tf.shape(input_left)[1]
        self.a_len = tf.shape(input_right)[1]
        self.batch_size = tf.shape(input_left)[0]
        Q = tf.reshape(input_left, [self.batch_size, self.q_len,
                       self.vector_size], name='Q')
        A = tf.reshape(
            input_right, [self.batch_size, self.a_len, self.vector_size], name='A')

        # [-1,vector_size] * [vector_size,vector_size] noting that * is matrix multiple
        first = tf.matmul(tf.reshape(Q, [self.batch_size * self.q_len, self.vector_size]), self.U)
        # [-1,vector_size]->[batch,q_len,vector_size]
        second_step = tf.reshape(first, [self.batch_size, self.q_len, self.vector_size])
        # [batch,q_len,vector_size]* [batch,vector,a_len]->[batch,q_len,a_len]

        A_transpose = tf.transpose(A, perm=[0, 2, 1])
        result = tf.matmul(second_step, A_transpose)
        print(second_step.get_shape().as_list())
        print(A_transpose.get_shape().as_list())
        G = tf.tanh(result)

        # column-wise pooling ,row-wise pooling
        # [batch,q_len,a_len]->[batch,1,a_len]
        row_pooling = tf.reduce_max(G, axis=1, keepdims = True, name='row_pooling')
        # [batch,q_len,a_len]->[batch,q_len,1]
        col_pooling = tf.reduce_max(G, axis=2, keepdims = True, name='col_pooling')

        attention_q = tf.nn.softmax(
            col_pooling, 1, name='attention_q')  # [batch,q_len,1]
        attention_a = tf.transpose(tf.nn.softmax(
            row_pooling, 2, name='attention_a'),perm = [0,2,1]) # [batch,a_len,1]

        R_q = tf.reduce_sum(tf.multiply(Q, attention_q), axis=1)
        R_a = tf.reduce_sum(tf.multiply(A, attention_a), axis=1)

        return R_q, R_a

    def wide_convolution(self, embedding):
        """
        docstring here wide convolution of the model
            :param self:
            :param embedding: embedding representation of the sentence
        """
        cnn_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.embedding_size, 1],
                    padding='SAME',
                    name="conv-{}".format(i)
            )
            h = tf.nn.relu(tf.nn.bias_add(
                conv, self.kernels[i][1]), name="relu-{}".format(i))
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs, 3)
        return cnn_reshaped

    def encode_sentence(self):
        """
        encode the sentence with cnn model
            :param self:
        """
        # pramaters of the attentive pooling
        self.vector_size = len(self.filter_sizes) * self.num_filters
        self.U = tf.Variable(tf.truncated_normal(
            shape=[self.vector_size, self.vector_size], stddev=0.01, name='U'))
        self.kernels = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="conv_w_filter_{}".format(i))
                conv_b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="conv_b_{}".format(i))
                self.kernels.append((conv_w, conv_b))

        q_emb = tf.expand_dims(self.q_embedding, -1)
        a_emb = tf.expand_dims(self.a_embedding, -1)
        a_neg_emb = tf.expand_dims(self.a_neg_embedding, -1)
        # convolution
        self.q_conv = self.wide_convolution(q_emb)
        self.a_conv = self.wide_convolution(a_emb)
        self.a_neg_conv = self.wide_convolution(a_neg_emb)

        # attentive pooling
        self.encode_q_pos, self.encode_a_pos= self.attentive_pooling(self.q_conv, self.a_conv)
        self.encode_q_neg, self.encode_a_neg= self.attentive_pooling(self.q_conv, self.a_neg_conv)
