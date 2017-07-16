# -*- coding:utf-8-*-
# 判断一个unicode是否是汉字
stopwords = { word.decode("utf-8") for word in open("../../model/chStopWordsSimple.txt").read().split()}
def is_chinese(uchar):         
    if u'\u4e00' <= uchar and uchar  <= u'\u9fff':
        return True
    else:
        return False
 
def is_number(uchar):
    if u'\u0030' <= uchar and uchar <= u'\u0039':
        return True
    else:
        return False
 
def is_alphabet(uchar):         
    if (u'\u0041' <= uchar<=u'\u005a') or (u'\u0061' <= uchar<=u'\u007a'):
        return True
    else:
        return False
def is_other(uchar):
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False
def processing_data():
    file = 'train.txt'
    with open(file) as f:
        for e,line in enumerate(f):
            splits = line.decode('utf-8').split('\t')
            q = splits[0]
            q = [word for word in q.split() if word not in stopwords]
            a = splits[1]
            # print a
            a = [word for word in a.split() if word not in stopwords]
            # print a
            print ' '.join(a)
            flag = splits[2]
            if e > 1000:
                break
if __name__ == '__main__':
    processing_data()
