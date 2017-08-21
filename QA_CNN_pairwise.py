import tensorflow as tf
import numpy as np


# model_type :apn or qacnn
class QA_CNN_extend(object):
    def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embedding_size,filter_sizes,num_filters,
        dropout_keep_prob = 1,learning_rate = 0.001,embeddings = None,l2_reg_lambda = 0.0,overlap_needed = False,trainable = True,extend_feature_dim = 10,pooling = 'attentive',position_needed = True,conv = 'narrow'):

        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.filter_sizes = filter_sizes
        self.l2_reg_lambda = l2_reg_lambda
        self.para = []
        self.extend_feature_dim = extend_feature_dim
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        self.overlap_needed = overlap_needed
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.trainable = trainable
        self.vocab_size = vocab_size
        self.pooling = pooling
        self.position_needed = position_needed
        self.conv = conv
        if self.overlap_needed:
            self.total_embedding_dim = embedding_size + extend_feature_dim
        else:
            self.total_embedding_dim = embedding_size
        #position embedding needed
        if self.position_needed:
            self.total_embedding_dim = self.total_embedding_dim + extend_feature_dim
        self.learning_rate = learning_rate
    def create_placeholder(self):
        print('Create placeholders')
        self.question = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'input_answer')
        self.answer_negative = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'input_right')
        self.q_pos_overlap = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_pos_feature_embed')
        self.q_neg_overlap = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_neg_feature_embed')
        self.a_pos_overlap = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_feature_embed')
        self.a_neg_overlap = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_neg_feature_embed')
        self.q_position = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_position_embed')
        self.a_pos_position = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_position_embed')
        self.a_neg_position = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_neg_postion_embed')
    def create_position(self):
        print 'add conv position'
        self.q_conv_position = tf.Variable(tf.ones([self.max_input_left,1]),name = 'q_conv_position')
        self.a_conv_position = tf.Variable(tf.ones([self.max_input_right,1]),name = 'a_conv_position')
    def add_embeddings(self):
        print 'add embeddings'
        if self.embeddings is not None:
            print "load embedding"
            W = tf.Variable(np.array(self.embeddings),name = "W" ,dtype="float32",trainable = self.trainable)
            
        else:
            print "random embedding"
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
        self.embedding_W = W
        self.overlap_W = tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
        # we suppose the max length of sentence is 300
        self.position_W = tf.Variable(tf.random_uniform([300,self.extend_feature_dim], -1.0, 1.0),name = 'W',trainable = True)
        # self.overlap_W = tf.Variable(a,name="W",trainable = True)
        self.para.append(self.embedding_W)
        self.para.append(self.overlap_W)
        self.para.append(self.position_W)
         #get embedding
        self.q_pos_embedding = self.concat_embedding(self.question,self.q_pos_overlap,self.q_position,self.q_conv_position)
        print self.q_pos_embedding
        self.q_neg_embedding = self.concat_embedding(self.question,self.q_neg_overlap,self.q_position,self.q_conv_position)
        self.a_pos_embedding = self.concat_embedding(self.answer, self.a_pos_overlap,self.a_pos_position,self.a_conv_position)
        self.a_neg_embedding = self.concat_embedding(self.answer_negative,self.a_neg_overlap,self.a_neg_position,self.a_conv_position)
    def convolution(self):
        print 'convolution:wide_convolution'
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_dim,1,self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        #convolution
        embeddings = [self.q_pos_embedding,self.q_neg_embedding,self.a_pos_embedding,self.a_neg_embedding]
        self.q_pos_feature_map,self.q_neg_feature_map,self.a_pos_feature_map,self.a_neg_feature_map = \
        [self.wide_convolution(embedding) for embedding in embeddings]
    def pooling_graph(self):
        print 'pooling: max pooling or attentive pooling'
        #pooling strategy
        if self.pooling == 'max':
            print self.pooling
            self.q_pos_pooling = tf.reshape(self.max_pooling(self.q_pos_feature_map,self.max_input_left),[-1,self.num_filters_total])
            self.q_neg_pooling = tf.reshape(self.max_pooling(self.q_neg_feature_map,self.max_input_left),[-1,self.num_filters_total])
            self.a_pos_pooling = tf.reshape(self.max_pooling(self.a_pos_feature_map,self.max_input_right),[-1,self.num_filters_total])
            self.a_neg_pooling = tf.reshape(self.max_pooling(self.a_neg_feature_map,self.max_input_right),[-1,self.num_filters_total])

        elif self.pooling == 'attentive':
            print self.pooling
            with tf.name_scope('attention'):    
                    self.U = tf.Variable(tf.truncated_normal(shape = [self.num_filters_total,self.num_filters_total],stddev = 0.01,name = 'U'))
                    self.para.append(self.U)
            self.q_pos_pooling,self.a_pos_pooling = self.attentive_pooling(self.q_pos_feature_map,self.a_pos_feature_map)
            self.q_neg_pooling,self.a_neg_pooling = self.attentive_pooling(self.q_neg_feature_map,self.a_neg_feature_map)
            # print self.q_pos_pooling
        else:
            print 'no implement'
            exit(0)  
    def create_loss(self):
        
        with tf.name_scope('score'):
            self.score12 = self.getCosine(self.q_pos_pooling,self.a_pos_pooling)
            self.score13 = self.getCosine(self.q_neg_pooling,self.a_neg_pooling)
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * l2_loss
        tf.summary.scalar('loss', self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)
    def create_op(self):
        self.global_step = tf.Variable(0, name="global_step", trainable = False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step = self.global_step)

    def concat_embedding(self,words_indice,overlap_indice,position_indice,conv_position):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        position_embedding = tf.nn.embedding_lookup(self.position_W,position_indice)
        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        if not self.overlap_needed :
            if not self.position_needed:
                all_embedding = embedded_chars_q
                # return tf.expand_dims(embedded_chars_q,-1)
            else:
                all_embedding = tf.concat([embedded_chars_q,position_embedding],2)
                # return tf.expand_dims(tf.concat([embedded_chars_q,position_embedding],2),-1)
        else:
            if not self.position_needed:
                all_embedding = tf.concat([embedded_chars_q,overlap_embedding_q],2)
                # return  tf.expand_dims(tf.concat([embedded_chars_q,overlap_embedding_q],2),-1)
            else:
                all_embedding = tf.concat([embedded_chars_q,overlap_embedding_q,position_embedding],2)
                # return tf.expand_dims(tf.concat([embedded_chars_q,overlap_embedding_q,position_embedding],2),-1)
        # all_embedding = tf.multiply(all_embedding,conv_position)
        return tf.expand_dims(all_embedding,-1)

    def max_pooling(self,conv,input_length):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, input_length, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
        return pooled
    def getCosine(self,q,a):
        pooled_flat_1 = tf.nn.dropout(q, self.dropout_keep_prob)
        pooled_flat_2 = tf.nn.dropout(a, self.dropout_keep_prob)
        
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1) 
        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores") 
        return score
    
    def attentive_pooling(self,input_left,input_right):
        Q = tf.reshape(input_left,[-1,self.max_input_left,len(self.filter_sizes) * self.num_filters],name = 'Q')
        A = tf.reshape(input_right,[-1,self.max_input_right,len(self.filter_sizes) * self.num_filters],name = 'A')
        # G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        # A,transpose_b = True),name = 'G')
        
        first = tf.matmul(tf.reshape(Q,[-1,len(self.filter_sizes) * self.num_filters]),self.U)
        print tf.reshape(Q,[-1,len(self.filter_sizes) * self.num_filters])
        print self.U
        second_step = tf.reshape(first,[-1,self.max_input_left,len(self.filter_sizes) * self.num_filters])
        result = tf.matmul(second_step,tf.nn.softmax(tf.transpose(A,perm = [0,2,1]),1))
        # print 'result',result
        G = tf.tanh(result)
        # G = result
        # column-wise pooling ,row-wise pooling
        row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
        col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

        self.attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
        print self.attention_q
        self.see = self.attention_q

        self.attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')
        R_q = tf.reshape(tf.matmul(Q,self.attention_q,transpose_a = 1),[-1,self.num_filters * len(self.filter_sizes)],name = 'R_q')
        R_a = tf.reshape(tf.matmul(self.attention_a,A),[-1,self.num_filters * len(self.filter_sizes)],name = 'R_a')

        return R_q,R_a
        
    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.total_embedding_dim, 1],
                    padding='SAME',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution_pooling(self):
        print 'narrow pooling'
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_dim,1,self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        embeddings = [self.q_pos_embedding,self.q_neg_embedding,self.a_pos_embedding,self.a_neg_embedding]
        self.q_pos_pooling,self.q_neg_pooling,self.a_pos_pooling,self.a_neg_pooling = [self.getFeatureMap(embedding,right = i / 2) for i,embedding in enumerate(embeddings) ]
    def getFeatureMap(self,embedding,right=True):
        if right == 1:
            max_length = self.max_input_right
        else:
            max_length = self.max_input_left
        pooled_outputs = []       
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")

            pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
            )
            pooled_outputs.append(pooled) 
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs,3), [-1, self.num_filters_total])  
        return pooled_reshape
    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_graph(self):
        self.create_placeholder()
        self.create_position()
        self.add_embeddings()
        if self.conv == 'narrow':
            self.narrow_convolution_pooling()
        else:
            self.convolution()
            self.pooling_graph()
        self.create_loss()
        self.create_op()
        self.merged = tf.summary.merge_all()

    
if __name__ == '__main__':
    cnn = QA_CNN_extend(max_input_left = 33,
        max_input_right = 40,
        batch_size = 3,
        vocab_size = 5000,
        embedding_size = 100,
        filter_sizes = [3,4,5],
        num_filters = 64, 
        dropout_keep_prob = 1.0,
        embeddings = None,
        l2_reg_lambda = 0.0,
        overlap_needed = False,
        trainable = True,
        extend_feature_dim = 10,
        position_needed = False,
        pooling = 'max',
        conv = 'wide')
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_x_3 = np.reshape(np.arange(3 * 40),[3,40])

    q_pos_embedding = np.ones((3,33))
    q_neg_embedding = np.ones((3,33))
    a_pos_embedding = np.ones((3,40))
    a_neg_embedding = np.ones((3,40)) 

    q_position = np.ones((3,33))
    a_pos_position = np.ones((3,40))
    a_neg_position = np.ones((3,40))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.answer_negative:input_x_3,
            # cnn.q_pos_overlap:q_pos_embedding,
            # cnn.q_neg_overlap:q_neg_embedding,
            # cnn.a_pos_overlap:a_pos_embedding,
            # cnn.a_neg_overlap:a_neg_embedding,
            # cnn.q_position:q_position,
            # cnn.a_pos_position:a_pos_position,
            # cnn.a_neg_position:a_neg_position
        }
        question,answer,score = sess.run([cnn.question,cnn.answer,cnn.score12],feed_dict)
        print question.shape,answer.shape
        print score


