import pynlpir
import jieba
import pandas as pd
pynlpir.open()
df = pd.DataFrame({'question':'', 'answer':'','flag':''},index = ['0'])
stopwords = { word.decode("utf-8") for word in open("../../model/chStopWordsSimple.txt").read().split()}
def type_line(line):
	try:
		segments = pynlpir.segment(line,pos_names='child')
		word_type = []
		words = []
		for seg in segments:
			if seg[1] == 'personal name' or seg[1] == 'transcribed personal name':
				words.append('NAME')
				word_type.append(seg[1])
			elif seg[1] == 'toponym' or seg[1] == 'locative word' or seg[1] == 'transcribed toponym':
				words.append('LOCATION')
				word_type.append(seg[1])
			elif seg[1] == 'time word':	
				words.append('TIME')
				word_type.append(seg[1])
			elif seg[1] == 'organization/group name':
				words.append('ORGANIZATION')
				word_type.append(seg[1])
			elif seg[1] == 'numeral':
				words.append('NUM')
				word_type.append(seg[1])
			else:
				words.append(seg[0])
	except:
		words = jieba.cut(str(sentence))
	
	line = ' '.join(words)
	return line
def cut_sentence(sentence):
	try:
		words= pynlpir.segment(sentence, pos_tagging=False)
		# return type_line(sentence)
	#print question
	except:
		words = jieba.cut(str(sentence))
	#print "$".join(words)
	# words=[word for word in words if word not in stopwords]
	return ' '.join(words)
def cut(row):
	question = row['question']
	answers = row['answer']
	flags = row['flag']
	question = cut(question)
	for answer, flag in zip(answers, flags):
		answer = cut(answer)
		temp = {'question':question, 'answer':answer, 'flag':flag}
		pd.concat([df, temp])


def splits():
	for filename in ["train_raw","test_raw","dev_raw"]:
		with open(filename+".txt") as f, open(filename[:-4]+".txt","w") as outf:
			for e,line in enumerate(f):
				if e % 100 == 0:
					print e
				splits = line.split('\t')
				# q = pynlpir.segment(splits[0],pos_tagging = False)
				# q = ' '.join([w for w in q if w not in stopwords]).encode('utf-8')
				# a = pynlpir.segment(splits[1],pos_tagging = False)
				# a = ' '.join([w for w in a if w not in stopwords]).encode('utf-8')
				try:
					q =  ' '.join(pynlpir.segment(splits[0],pos_tagging = False)).encode('utf-8')	
				except Exception as e:
					q = ' '.join(jieba.cut(str(splits[0]))).encode('utf-8')	
				try:
					a =  ' '.join(pynlpir.segment(splits[1],pos_tagging = False)).encode('utf-8')
				except Exception as e:
					a = ' '.join(jieba.cut(str(splits[1]))).encode('utf-8')
				flag = splits[2].encode('utf-8')
				outf.write(q + '\t' + a + '\t' + flag)
def dataPrepare(orginal):
	df = orginal.copy()	
	df["question"]=df["question"].apply(cut_sentence)
	df["answer"]=df["answer"].apply(cut_sentence)
	return df
def splits2():
	data_dir = 'raw_data/'
	test_file = data_dir + "test_raw.txt"
	train_file = data_dir + "train_raw.txt"
	dev_file = data_dir + "dev_raw.txt"
	train = pd.read_csv(train_file,header = None,sep = "\t",names=["question","answer","flag"],quoting =3)
	test = pd.read_csv(test_file,header = None,sep = "\t",names=["flag","question","answer"],quoting =3)
	dev = pd.read_csv(dev_file,header = None,sep = '\t',names = ["question","answer","flag"],quoting = 3)
	# print len(train['question'].unique())
	train = dataPrepare(train)
	# print len(train['question'].unique())
	test = dataPrepare(test)
	dev = dataPrepare(dev)
	train = train[['question','answer','flag']]
	test = test[['question','answer','flag']]
	dev = dev[['question','answer','flag']]
	
	train.to_csv('train.txt',header = None,sep = '\t',index = None,encoding='utf-8')
	test.to_csv('test.txt',header = None,sep = '\t',index = None,encoding='utf-8')
	dev.to_csv('dev.txt',header = None,sep = '\t',index = None,encoding='utf-8')
def submit_data():
	submit_file = 'submit_raw.txt'
	submit_data = pd.read_csv(submit_file,header = None,sep = '\t',names = ['question','answer'],quoting = 3)
	submit_data = dataPrepare(submit_data)
	submit_data.to_csv('submit.txt',header = None,sep = '\t',index = None,encoding = 'utf-8')
def test_data():
	test_file = 'dbqa.txt'
	test_data = pd.read_csv(test_file,header = None,sep = '\t',names = ['flag','question','answer'],quoting = 3)
	test_data = dataPrepare(test_data)
	convert_data = test_data[['question','answer','flag']]
	convert_data.to_csv('dbqa_submit.txt',header = None,sep = '\t',index = None,encoding = 'utf-8')
	
if __name__ == '__main__':
	# test_data()
	splits2()
	# submit_data()
	# splits2()
			# if e % 100 == 0
			# 	print e
			
			# print splits
			# try:
			# 	q =  pynlpir.segment(line[0], pos_tagging = False)
			# 	print q
			# 	a = pynlpir.segment(line[1],pos_tagging = False)
			# 	f = line[2][:-2]
			# 	outf.write(q + '\t' + a + '\t' + f)
			# 	# print words
			# 	# words = pynlpir.segment(line, pos_tagging=False)
			# except:
			# 	print 'segment error'
			# 	words = jieba.cut(str(line))
			# 	outf.write(" ".join([item.encode("utf-8") for item in words]))
# 
# f = open('train.txt','r')
# for e,line in enumerate(f):
# 	print e,line
# 	if e > 10:
# 		break