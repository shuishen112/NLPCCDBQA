'''
@Description: 
@Author: zhansu
@Date: 2019-06-28 20:14:28
@LastEditTime: 2019-07-23 21:00:37
@LastEditors: Please set LastEditors
'''
from tensorflow import flags
import tensorflow as tf
from config import args
import helper
import time
import datetime
import os
from models.cnn_model import Attentive_CNN
import numpy as np
import evaluation
import sys
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(os.getcwd())

now = int(time.time())
timeArray = time.localtime(now)
log_filename = "log/" + time.strftime("%Y%m%d", timeArray)
if not os.path.exists(log_filename):
    os.makedirs(log_filename)

program = os.path.basename('QA')
logger = logging.getLogger(program)

logging.basicConfig(format = '%(asctime)s: %(levelname)s: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_filename+'/{}_qa.log'.format(time.strftime("%H%M", timeArray)), filemode='w')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


opts = args.flag_values_dict()
for item in opts:
    logger.info('{} : {}'.format(item, opts[item]))

logger.info('load data ...........')
train, test, dev = helper.load_train_file(
    opts['data_dir'], filter=args.clean)

q_max_sent_length = max(map(lambda x: len(x), train['question'].str.split()))
a_max_sent_length = max(map(lambda x: len(x), train['answer'].str.split()))

alphabet = helper.get_alphabet([train, test, dev])
logger.info('the number of words :%d ' % len(alphabet))

embedding = helper.get_embedding(
    alphabet, opts['embedding_file'], embedding_size=opts['embedding_size'])

opts["embeddings"] = embedding
opts["vocab_size"] = len(alphabet)
opts["max_input_right"] = a_max_sent_length
opts["max_input_left"] = q_max_sent_length
opts["filter_sizes"] = list(map(int, args.filter_sizes.split(",")))

with tf.Graph().as_default():

    model = Attentive_CNN(opts)
    model._model_stats()
    for i in range(args.num_epoches):
        data_gen = helper.batch_iter(train, args.batch_size,alphabet,shuffle=True,q_len=q_max_sent_length,a_len=a_max_sent_length )
        model.train(data_gen,i)

        test_datas = helper.batch_iter(
            test, args.batch_size,alphabet,q_len=q_max_sent_length,a_len=a_max_sent_length )

        test['score'] = model.predict(test_datas)
        map_, mrr_= evaluation.evaluationBypandas(test, test['score'].to_list())
        df_group = test.groupby('question').filter(evaluation.mrr_metric_filter)
        df_group[['question','answer','flag','score']].to_csv('badcase',sep = '\t',index = None)
        logger.info('map:{}--mrr:{}'.format(map_, mrr_))
        print('map:{}--mrr:{}'.format(map_, mrr_))
