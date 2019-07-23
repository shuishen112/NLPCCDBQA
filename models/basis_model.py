'''
@Description: this is the basis model
@Author: zhansu
@Date: 2019-07-02 20:58:41
@LastEditTime: 2019-07-23 21:21:17
@LastEditors: Please set LastEditors
'''
# coding:utf-8

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from tensorflow.contrib import rnn
import models.blocks as blocks
import datetime
from functools import reduce
import abc
import sys
sys.path.append('../')
# tf.set_random_set()


class Model(object):

    def __init__(self, opt):
        """
        initialize the model by the para
        pair_wise model
            :param self: 
            :param opt: para of the model in the config
        """
        for key, value in opt.items():
            self.__setattr__(key, value)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.build_graph()
        # summary
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.summaries_dir + '/train',
                                                  self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.summaries_dir + '/test')
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # whether debug the code
        if self.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

    def build_graph(self):
        """
        build the graph of the model
            :param self: 
        """
        self.create_placeholder()
        self.add_embeddings()
        self.encode_sentence()
        self.create_loss()
        self.create_op()

    def create_placeholder(self):

        print(('Create placeholders'))
        # he length of the sentence is varied according to the batch,so the None,None
        self.question = tf.placeholder(
            tf.int32, [None, None], name='input_question')

        self.answer = tf.placeholder(
            tf.int32, [None, None], name='input_answer')
        self.answer_negative = tf.placeholder(
            tf.int32, [None, None], name='input_right')

        self.batch_size = tf.shape(self.question)[0]
        self.q_len, self.q_mask = blocks.length(self.question)
        self.a_len, self.a_mask = blocks.length(self.answer)
        self.a_neg_len, self.a_neg_mask = blocks.length(self.answer_negative)
        self.dropout_keep_prob_holder = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

    def add_embeddings(self):
        print('add embeddings')

        self.embedding_w = tf.Variable(np.array(self.embeddings), name="embedding",
                                       dtype="float32", trainable=self.trainable)

        self.q_embedding = tf.nn.embedding_lookup(
            self.embedding_w, self.question, name="q_embedding")
        self.a_embedding = tf.nn.embedding_lookup(
            self.embedding_w, self.answer, name="a_embedding")
        self.a_neg_embedding = tf.nn.embedding_lookup(
            self.embedding_w, self.answer_negative, name="a_neg_embedding")

    def get_cosine(self, q, a, name):
        """
        docstring here
            :param self: 
            :param q: [batch, vector_size]
            :param a: [batch, vector_size]
        """
        if self.dropout_keep_prob_holder != 1.0:

            pooled_flat_1 = tf.nn.dropout(q, self.dropout_keep_prob_holder)
            pooled_flat_2 = tf.nn.dropout(a, self.dropout_keep_prob_holder)

            cosine = tf.div(
                tf.reduce_sum(pooled_flat_1*pooled_flat_2, 1),
                tf.sqrt(tf.reduce_sum(pooled_flat_1*pooled_flat_1, 1)) *
                tf.sqrt(tf.reduce_sum(pooled_flat_2*pooled_flat_2, 1)) + 1e-8,
                name="cosine")

            return cosine

            # q_normalize = tf.nn.l2_normalize(pooled_flat_1, dim=1)
            # a_normalize = tf.nn.l2_normalize(pooled_flat_2, dim=1)
        else:
            #     q_normalize = tf.nn.l2_normalize(q, dim=1)
            #     a_normalize = tf.nn.l2_normalize(a, dim=1)

            cosine = tf.div(
                tf.reduce_sum(q*a, 1),
                tf.sqrt(tf.reduce_sum(q*q, 1)) *
                tf.sqrt(tf.reduce_sum(a*a, 1)) + 1e-8,
                name="cosine")

        # score = tf.reduce_sum(tf.multiply(q_normalize, a_normalize), 1)

            return cosine

    def create_op(self):

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(
            self.grads_and_vars, global_step=self.global_step)

    def create_loss(self):
        """
        calculate the loss, noting that we don't use the l2_regularizer
            :param self: 
        """
        with tf.name_scope('score'):
            self.score12 = self.get_cosine(
                self.encode_q_pos, self.encode_a_pos, name="pos_score")
            self.score13 = self.get_cosine(
                self.encode_q_neg, self.encode_a_neg, name="neg_score")

        with tf.name_scope("loss"):
            l2_loss = 0.0
            for para in tf.trainable_variables():
                l2_loss += tf.nn.l2_loss(para)
            self.losses = tf.maximum(0.0, tf.subtract(
                0.05, tf.subtract(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * l2_loss

        tf.summary.scalar('loss', self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self, data_batch, i):
        """
        thain the model 
            :param self: 
            :param data_batch: train_dataset databatch
        """
        for data in data_batch:
            question,pos_answer,neg_answer = zip(*data)
            feed_dict = {
                self.question: question,
                self.answer: pos_answer,
                self.answer_negative:neg_answer,
                self.dropout_keep_prob_holder: self.dropout_keep_prob
            }
            _, summary, step, loss, accuracy, score12, score13 = self.sess.run(
                [self.train_op, self.merged, self.global_step, self.loss,
                    self.accuracy, self.score12, self.score13],
                feed_dict)
            self.train_writer.add_summary(summary, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: epoch:{},step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g},score{}".format(
                time_str, i, step, loss, accuracy, np.mean(score12), np.mean(score13), np.mean(score12)))

    def predict(self, data_batch):
        """
        predict the test_dataset
            :param self: 
            :param data_batch: test_dataset data_batch
        """
        scores = []
        for e, data in enumerate(data_batch):

            question,answer = zip(*data)
            feed_dict = {
                self.question: question,
                self.answer:answer,
                self.dropout_keep_prob_holder: 1.0
            }
            score = self.sess.run(
                self.score12, feed_dict)
            # self.test_writer.add_summary(summary, e)
            scores.extend(score)
        return scores

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @abc.abstractmethod
    def encode_sentence(self):
        """
        the method is the implemented by the subclass
            :param self: 
        """

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())
        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" %
                  (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v)
                                            for v in tf.trainable_variables())))
