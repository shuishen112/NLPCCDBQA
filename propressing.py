'''
@Description: 进行数据分析，关注数据细节
@Author: zhansu
@Date: 2019-07-05 17:26:53
@LastEditTime: 2019-07-23 15:52:32
@LastEditors: Please set LastEditors
'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
pal = sns.color_palette()
print(os.getcwd())

df_train = pd.read_csv('data/nlpcc/train.txt', sep='\t',
                       names=['question', 'answer', 'flag'], quoting=3)

print(df_train['flag'].head())

# 基础分析
print(df_train.info())
print(df_train.shape)
df_train.groupby('flag')['question'].count().plot.bar()
print("dataset size:{}".format(len(df_train)))
print("positive sample rate:{}%".format(
    round(df_train['flag'].mean() * 100, 2)))
print('question pairs:{}'.format(len(df_train['question'].unique())))

# 文本分析
df_test = pd.read_csv('data/nlpcc/test.txt', sep='\t',
                      names=['question', 'answer', 'flag'], quoting=3)

train_qs = pd.Series(
    df_train['question'].tolist() + df_train['answer'].tolist())
test_qs = pd.Series(df_test['question'].tolist() + df_test['answer'].tolist())
dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))
print('mean-train:{} std-train:{} max-train:{} mean-test:{} std-test:{} max-test:{}'.format(dist_train.mean(),
                                                                                            dist_train.std(),
                                                                                            dist_train.max(),
                                                                                            dist_test.mean(),
                                                                                            dist_test.std(),
                                                                                            dist_test.max()))

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=40, range=[0, 40],
         color=pal[2], normed=True, label='train')
plt.hist(dist_test, bins=40, range=[
         0, 40], color=pal[1], normed=True, alpha=0.5, label='test')
plt.title('Normalised histogram of character count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.show()

# 语义分析
