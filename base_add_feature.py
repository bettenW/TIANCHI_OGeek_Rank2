
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
import jieba, os, Levenshtein, time
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from utility import read_file, lcseque_lens, lcsubstr_lens, find_longest_prefix, printlog
from sklearn import preprocessing
import numpy as np
from xpinyin import Pinyin


# In[2]:

print('run base_add_feature')
# 配置信息
is_print_output = True
all_start_time = time.time()


# In[3]:


since = time.time()
# 读入数据
train_data = read_file('Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt')
val_data = read_file('Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt')
test_data = read_file('Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt', True)

# 拼接数据一起做特征
not_zip_all_data = pd.concat((train_data, val_data, test_data), axis = 0, ignore_index = True, sort = False)

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[4]:


since = time.time()
# 修正为空的query_prediction
not_zip_all_data.loc[not_zip_all_data.query_prediction == '', 'query_prediction'] = '{}'
# 修正label为int
not_zip_all_data['label'] = not_zip_all_data.label.astype('int')
# 保存要丢弃掉的列
drop_feature = []

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[5]:


since = time.time()
# 根据prefix query_prediction title tag来merge, query_prediction需要编码处理
encoder = preprocessing.LabelEncoder()
not_zip_all_data['diction_label'] = encoder.fit_transform(not_zip_all_data.query_prediction)

# 去除重复算非统计量特征
all_data = not_zip_all_data.drop('label', axis = 1).drop_duplicates().reset_index(drop = True)

drop_feature.append('diction_label')

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[9]:


all_data['prefix_contains_tag'] = all_data.groupby('prefix').tag.transform(lambda x: " ".join(x))


# In[12]:


since = time.time()
# 拼接回原数据
all_data = all_data.drop('query_prediction', axis = 1)
not_zip_all_data = pd.merge(not_zip_all_data, all_data, how = 'left', on = ['prefix', 'title', 'tag', 'diction_label'])

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[14]:


since = time.time()
cv = CountVectorizer() #, max_df = 0.03
prefix_contains_tag_matrix = cv.fit_transform(not_zip_all_data.prefix_contains_tag)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[17]:


since = time.time()
# 保存数据
sparse.save_npz('Demo/Cases/prefix_contains_tag.npz', prefix_contains_tag_matrix)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[18]:


time_elapsed = time.time() - all_start_time
print('final complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间

