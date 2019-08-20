
# coding: utf-8

# In[1]:


import jieba, os, Levenshtein, random, copy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from scipy import sparse
import pandas as pd
from tqdm import tqdm
import jieba, os, Levenshtein, time
from utility import *
from sklearn import preprocessing
import numpy as np 
from xpinyin import Pinyin
#自定义词典
# jieba.load_userdict("./coal_dict.txt")


# In[2]:
print('run count_vectore_feature')

since = time.time()
# 读入数据
train_data = read_file('Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt')
val_data = read_file('Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt')
test_data = read_file('Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt', True)
    
# 拼接数据一起做特征
all_data = pd.concat((train_data, val_data, test_data), axis = 0, ignore_index = True, sort = False)

# 修正query_prediction为空
all_data.loc[all_data.query_prediction == '', 'query_prediction'] = '{}'
# 转换label的数据类型
all_data['label'] = all_data.label.astype('int')
# all_data['prefix'] = all_data.prefix.apply(lambda x: x.replace(' ', '').lower())

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[3]:


all_data.shape


# In[4]:


since = time.time()
# 压缩数据并提取max_query_prediction_keys特征
tag_encoder = preprocessing.LabelEncoder()
all_data['diction_label'] = tag_encoder.fit_transform(all_data.query_prediction)

zip_data = all_data.drop('label', axis = 1).drop_duplicates().reset_index(drop = True)

def str_to_dict(dict_str):
    """str convert to dict"""
    my_dict = eval(dict_str)
    keys, values = my_dict.keys(), my_dict.values()
    my_dict = dict(zip(keys,list(map(lambda x: float(x), values))))
    return my_dict
zip_data['query_prediction'] = zip_data.query_prediction.apply(lambda x: str_to_dict(x))
zip_data['max_query_prediction_keys'] = zip_data.query_prediction.apply(lambda x: '' if x == {} else max(x, key = x.get))

zip_data['prefix_jieba'] = zip_data.prefix.apply(lambda x: " ".join(jieba.cut(x, cut_all = False)))
zip_data['title_jieba'] = zip_data.title.apply(lambda x: " ".join(jieba.cut(x, cut_all = False)))

def query_prediction_jieba(query_dict):
    query_keys = list(query_dict.keys())
    res = []
    for keys in query_keys:
        res.append(" ".join(jieba.cut(keys, cut_all = False)))
    return " ".join(res)
zip_data['query_prediction_jieba'] = zip_data.query_prediction.apply(lambda x: query_prediction_jieba(x))


time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[5]:


since = time.time()
# 拼接回原数据
zip_data = zip_data.drop('query_prediction', axis = 1)
all_data = pd.merge(all_data, zip_data, how = 'left', on = ['prefix', 'title', 'tag', 'diction_label'])

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[6]:


since = time.time()

cv = CountVectorizer(token_pattern = '\w+', min_df = 2, ngram_range = (1, 2)) #, max_df = 0.03
prefix_matrix = cv.fit_transform(all_data.prefix_jieba)

cv = CountVectorizer(token_pattern = '\w+', min_df = 2, ngram_range = (1, 2)) #, max_df = 0.03
title_matrix = cv.fit_transform(all_data.title_jieba)

cv = CountVectorizer(token_pattern = '\w+', min_df = 2, ngram_range = (1, 2)) #, max_df = 0.03
query_matrix = cv.fit_transform(all_data.query_prediction_jieba)

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[7]:


all_cv = sparse.hstack((prefix_matrix, title_matrix, query_matrix))
# test_cv = sparse.hstack((title_cv_test, prefix_test))


# In[ ]:


sparse.save_npz('Demo/Cases/all_cv2.npz', all_cv)


# In[16]:


all_cv.shape

