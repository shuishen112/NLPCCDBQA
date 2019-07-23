# -*- coding:utf-8-*-
import numpy as np
import random,os,math
import pandas as pd
import sklearn
import time
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import evaluation
import string
import jieba
from nltk import stem
from tqdm import tqdm
import chardet
import re
import config
import logging
from functools import wraps

# stopwords = { word.decode("utf-8") for word in open("model/chStopWordsSimple.txt").read().split()}
# ner_dict = pickle.load(open('ner_dict'))

#print( tf.__version__)
def log_time_delta(func):
	@wraps(func)
	def _deco(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		end = time.time()
		delta = end - start
		print( "%s runed %.2f seconds"% (func.__name__,delta))
		return ret
	return _deco
def remove_the_unanswered_sample(df):
	"""
	clean the dataset
			:param df: dataframe
	"""
	counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
	questions_have_correct = counter[counter > 0].index
	counter = df.groupby("question").apply(
		lambda group: sum(group["flag"] == 0))
	questions_have_uncorrect = counter[counter > 0].index
	counter = df.groupby("question").apply(lambda group: len(group["flag"]))
	questions_multi = counter[counter > 1].index

	return df[df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()

def load_train_file(data_dir, filter=False):
	"""
	load the dataset
			:param data_dir: the data_dir
			:param filter=False: whether clean the dataset
	"""
	train_df = pd.read_csv(os.path.join(data_dir, 'train.txt'), header=None, sep='\t', names=[
		'question', 'answer', 'flag'], quoting=3).fillna('')
	if filter:
		train_df = remove_the_unanswered_sample(train_df)
	dev_df = pd.read_csv(os.path.join(data_dir, 'dev.txt'), header=None, sep='\t', names=[
		'question', 'answer', 'flag'], quoting=3).fillna('')
	if filter:
		dev_df = remove_the_unanswered_sample(dev_df)
	test_df = pd.read_csv(os.path.join(data_dir, 'test.txt'), header=None, sep='\t', names=[
		'question', 'answer', 'flag'], quoting=3).fillna('')
	if filter:
		test_df = remove_the_unanswered_sample(test_df)
	return train_df, test_df, test_df

def cut(sentence):
	"""
	split the sentence to tokens
			:param sentence: raw sentence
	"""
	tokens = sentence.split()

	return tokens

def get_alphabet(corpuses):
	"""
	obtain the dict
			:param corpuses: 
	"""
	word_counter = Counter()

	for corpus in corpuses:
		for texts in [corpus["question"].unique(), corpus["answer"]]:
			for sentence in texts:
				tokens = cut(sentence)
				for token in tokens:
					word_counter[token] += 1
	print("there are {} words in dict".format(len(word_counter)))
	logging.info("there are {} words in dict".format(len(word_counter)))
	word_dict = {word: e + 2 for e, word in enumerate(list(word_counter))}
	word_dict['UNK'] = 1
	word_dict['<PAD>'] = 0

	return word_dict

def get_embedding(alphabet, filename="", embedding_size=100):
	embedding = np.random.rand(len(alphabet), embedding_size)
	if filename is None:
		return embedding
	with open(filename, encoding='utf-8') as f:
		i = 0
		for line in f:
			i += 1
			if i % 100000 == 0:
				print('epch %d' % i)
			items = line.strip().split(' ')
			if len(items) == 2:
				vocab_size, embedding_size = items[0], items[1]
				print((vocab_size, embedding_size))
			else:
				word = items[0]
				if word in alphabet:
					embedding[alphabet[word]] = items[1:]

	print('done')
	return embedding


def convert_to_word_ids(sentence,alphabet,max_len = 40):
	"""
	docstring here
		:param sentence: 
		:param alphabet: 
		:param max_len=40: 
	"""
	indices = []
	tokens = cut(sentence)
	
	for word in tokens:
		if word in alphabet:
			indices.append(alphabet[word])
		else:
			continue
	result = indices + [alphabet['<PAD>']] * (max_len - len(indices))

	return result[:max_len]
def gen_with_pair_train(df, alphabet, q_len,a_len):
	pairs = []
	for question in df['question'].unique():
    		
		
		group = df[df['question'] == question]
		pos_group = group[group['flag'] == 1] # positive answer
		neg_group = group[group['flag'] == 0]
		neg_group = neg_group.reset_index()
	
		question_indice = convert_to_word_ids(question,alphabet,max_len = q_len)

		negtive_pool_index = range(len(neg_group))

		if len(neg_group) > 0:
			for pos in pos_group['answer']:
				neg_index = np.random.choice(negtive_pool_index)
				neg = neg_group.loc[neg_index]['answer']

				positive_answer_indice = convert_to_word_ids(pos,alphabet,a_len)
				negative_answer_indice = convert_to_word_ids(neg,alphabet,a_len)
				pairs.append((question_indice,positive_answer_indice,negative_answer_indice))
	return pairs

def gen_with_pair_test(df,alphabet,q_len,a_len):
	pairs = []
	for _,row in df.iterrows():
		question_indice = convert_to_word_ids(row['question'],alphabet,max_len=q_len)
		answer_indice = convert_to_word_ids(row['answer'],alphabet,max_len = a_len)
		pairs.append((question_indice,answer_indice))

	return pairs
def batch_iter(data, batch_size, alphabet,shuffle = False,q_len = 33,a_len = 33):
	if shuffle:
		data = gen_with_pair_train(
			data, alphabet,q_len,a_len )
	else:
		data = gen_with_pair_test(data,alphabet,q_len,a_len)
	data = np.array(data)
	data_size = len(data)

	if shuffle:
		shuffle_indice = np.random.permutation(np.arange(data_size))
		data = data[shuffle_indice]

	num_batch = int((data_size - 1) / float(batch_size)) + 1

	for i in range(num_batch):
	  start_index = i * batch_size
	  end_index = min((i + 1) * batch_size, data_size)

	  yield data[start_index:end_index]

@log_time_delta
def get_overlap_dict(df,alphabet,q_len = 40,a_len = 40):
	d = dict()
	for question in df['question'].unique():
		group = df[df['question'] == question]
		answers = group['answer']
		for ans in answers:
			q_overlap,a_overlap = overlap_index(question,ans,q_len,a_len)
			d[(question,ans)] = (q_overlap,a_overlap)
	return d
# calculate the overlap_index
def overlap_index(question,answer,q_len,a_len,stopwords = []):
	qset = set(cut(question))
	aset = set(cut(answer))

	q_index = np.zeros(q_len)
	a_index = np.zeros(a_len)

	overlap = qset.intersection(aset)
	for i,q in enumerate(cut(question)[:q_len]):
		value = 1
		if q in overlap:
			value = 2
		q_index[i] = value
	for i,a in enumerate(cut(answer)[:a_len]):
		value = 1
		if a in overlap:
			value = 2
		a_index[i] = value
	return q_index,a_index
def position_index(sentence,length):
	index = np.zeros(length)

	raw_len = len(cut(sentence))
	index[:min(raw_len,length)] = range(1,min(raw_len + 1,length + 1))
	# print index
	return index
def transform(flag):
	if flag == 1:
		return [0,1]
	else:
		return [1,0]
@log_time_delta
def batch_gen_with_single(df,alphabet,batch_size = 10,q_len = 33,a_len = 40,overlap_dict = None):
	pairs=[]
	for index,row in df.iterrows():
		quetion = encode_to_split(row["question"],alphabet,max_sentence = q_len)
		answer = encode_to_split(row["answer"],alphabet,max_sentence = a_len)
		if overlap_dict:
			q_pos_overlap,a_pos_overlap = overlap_index(row["question"],row["answer"],q_len,a_len)
		else:
			q_pos_overlap,a_pos_overlap = overlap_dict[(row["question"],row["answer"])]

		q_position = position_index(row['question'],q_len)
		a_pos_position = position_index(row['answer'],a_len)
		pairs.append((quetion,answer,q_pos_overlap,a_pos_overlap,q_position,a_pos_position))
	# n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
	# n_batches = int(len(pairs)*1.0/batch_size)
	# # pairs = sklearn.utils.shuffle(pairs,random_state =132)
	# for i in range(0,n_batches):
	#     batch = pairs[i*batch_size:(i+1) * batch_size]
	num_batches_per_epoch = int((len(pairs)-1)/ batch_size) + 1
	for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, len(pairs))
			batch = pairs[start_index:end_index]
			yield [[pair[j] for pair in batch]  for j in range(6)]
	# batch= pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
	# yield [[pair[i] for pair in batch]  for i in range(6)]
def overlap_visualize():
	train,test,dev = load("nlpcc",filter = False)

	test = test.reindex(np.random.permutation(test.index))
	df = train
	df['qlen'] = df['question'].str.len()
	df['alen'] = df['answer'].str.len()

	df['q_n_words'] = df['question'].apply(lambda row:len(row.split(' ')))
	df['a_n_words'] = df['answer'].apply(lambda row:len(row.split(' ')))

	def normalized_word_share(row):
		w1 = set(map(lambda word: word.lower().strip(), row['question'].split(" ")))
		w2 = set(map(lambda word: word.lower().strip(), row['answer'].split(" ")))    
		return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
	def word_overlap(row):
		w1 = set(map(lambda word: word.lower().strip(), row['question'].split(" ")))
		w2 = set(map(lambda word: word.lower().strip(), row['answer'].split(" ")))
		return w1.intersection(w2)
	df['word_share'] = df.apply(normalized_word_share, axis=1)
	plt.figure(figsize=(12, 8))
	plt.subplot(1,2,1)
	sns.violinplot(x = 'flag', y = 'word_share', data = df[0:50000],hue = 'flag')
	plt.subplot(1,2,2)
	# sns.distplot(df[df['flag'] == 1.0]['word_share'][0:10000], color = 'green',label = 'not match')
	# sns.distplot(df[df['flag'] == 0.0]['word_share'][0:10000], color = 'blue',label = 'match')

	# plt.figure(figsize=(15, 5))
	train_word_match = df.apply(normalized_word_share, axis=1, raw=True)
	plt.hist(train_word_match[df['flag'] == 0], bins=20, normed=True, label='flag 0')
	plt.hist(train_word_match[df['flag'] == 1], bins=20, normed=True, alpha=0.7, label='flag 1')
	plt.legend()
	plt.title('Label distribution over word_match_share', fontsize=15)
	plt.xlabel('word_match_share', fontsize=15)

	# train_qs = pd.Series(train['question'].tolist() + train['answer'].tolist())
	# print train_qs
	plt.show('hold')
def dns_sample(df,alphabet,q_len,a_len,sess,model,batch_size,neg_sample_num = 10):
	samples = []
	count = 0
	pool_answers = df[df.flag == 1]['answer'].tolist()
	# pool_answers = df[df['flag'] == 0]['answer'].tolist()
	for question in df['question'].unique():
		group = df[df['question'] == question]
		pos_answers = group[df["flag"]==1]["answer"].tolist()
		# pos_answers_exclude = list(set(pool_answers).difference(set(pos_answers)))
		neg_answers = group[df["flag"]==0]["answer"].tolist()
		question_indices = encode_to_split(question,alphabet,max_sentence = q_len)
		for pos in pos_answers:
			# negtive sample
			neg_pool = []
			if len(neg_answers) > 0:
				# neg_exc = list(np.random.choice(pos_answers_exclude,size = 100 - len(neg_answers)))
				neg_answers_sample = neg_answers
				# neg_answers = neg_a
				# print 'neg_tive answer:{}'.format(len(neg_answers))
				for neg in neg_answers_sample:
					neg_pool.append(encode_to_split(neg,alphabet,max_sentence = a_len))
				input_x_1 = [question_indices] * len(neg_answers_sample)
				input_x_2 = [encode_to_split(pos,alphabet,max_sentence = a_len)] * len(neg_answers_sample)
				input_x_3 = neg_pool
				feed_dict = {
					model.question: input_x_1,
					model.answer: input_x_2,
					model.answer_negative:input_x_3 
				}
				predicted = sess.run(model.score13,feed_dict)
				# find the max score
				index = np.argmax(predicted)
				# print len(neg_answers)
				# print 'index:{}'.format(index)
				# if len(neg_answers)>1:
				#     print neg_answers[1]
				samples.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),input_x_3[index]))      
				count += 1
				if count % 100 == 0:
					print ('samples load:{}'.format(count))
	print ('samples finishted len samples:{}'.format(len(samples)))
	return samples
@log_time_delta
def batch_gen_with_pair_dns(samples,batch_size,epoches=1):
	# n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
	n_batches = int(len(samples) * 1.0 / batch_size)
	for j in range(epoches):
		pairs = sklearn.utils.shuffle(samples,random_state =132)
		for i in range(0,n_batches):
			batch = pairs[i*batch_size:(i+1) * batch_size]
			yield [[pair[i] for pair in batch]  for i in range(3)]

def data_processing():
	train,test,dev = load('nlpcc',filter = False)
	q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
	a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
	q_len = map(lambda x:len(x),train['question'].str.split())
	a_len = map(lambda x:len(x),train['answer'].str.split())
	print('Total number of unique question:{}'.format(len(train['question'].unique())))
	print('Total number of question pairs for training: {}'.format(len(train)))
	print('Total number of question pairs for test: {}'.format(len(test)))
	print('Total number of question pairs for dev: {}'.format(len(dev)))
	print('Duplicate pairs: {}%'.format(round(train['flag'].mean()*100, 2)))
	print(len(train['question'].unique()))

	#text analysis
	train_qs = pd.Series(train['answer'].tolist())
	test_qs = pd.Series(test['answer'].tolist())
	dev_qs = pd.Series(dev['answer'].tolist())

	dist_train = train_qs.apply(lambda x:len(x.split(' ')))
	dist_test = test_qs.apply(lambda x:len(x.split(' ')))
	dist_dev = dev_qs.apply(lambda x:len(x.split(' ')))
	pal = sns.color_palette()
	plt.figure(figsize=(15, 10))
	plt.hist(dist_train, bins = 200, range=[0, 200], color=pal[2], normed = True, label='train')
	plt.hist(dist_dev, bins = 200, range=[0, 200], color=pal[3], normed = True, alpha = 0.5, label='test1')
	plt.hist(dist_test, bins = 200, range=[0, 200], color=pal[1], normed = True, alpha = 0.5, label='test2')
	
	plt.title('Normalised histogram of tokens count in answers', fontsize = 15)
	plt.legend()
	plt.xlabel('Number of words', fontsize = 15)
	plt.ylabel('Probability', fontsize = 15)

	print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
						  dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
	plt.show('hard')

	qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
	who = np.mean(train_qs.apply(lambda x:'Who' in x))
	where = np.mean(train_qs.apply(lambda x:'Where' in x))
	how_many = np.mean(train_qs.apply(lambda x:'How many' in x))
	fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
	capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
	capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
	numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))
	print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
	print('Questions with [Who] tags: {:.2f}%'.format(who * 100))
	print('Questions with [where] tags: {:.2f}%'.format(where * 100))
	print('Questions with [How many] tags:{:.2f}%'.format(how_many * 100))
	print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
	print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
	print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
	print('Questions with numbers: {:.2f}%'.format(numbers * 100))