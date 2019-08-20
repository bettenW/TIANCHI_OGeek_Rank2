
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

print('run base_feature')
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


# In[6]:


since = time.time()
# 解析抽取字典
def str_to_dict(dict_str):
    """str convert to dict"""
    my_dict = eval(dict_str)
    keys, values = my_dict.keys(), my_dict.values()
    my_dict = dict(zip(keys,list(map(lambda x: float(x), values))))
    return my_dict
all_data['query_prediction'] = all_data.query_prediction.apply(lambda x: str_to_dict(x))
# 取出最大value对应的keys
all_data['max_query_prediction_keys'] = all_data.query_prediction.apply(lambda x: '' if x == {} else max(x, key = x.get))
# 取出字典的keys和values
all_data['query_prediction_keys'] = all_data.query_prediction.apply(lambda x: list(x.keys()))
all_data['query_prediction_values'] = all_data.query_prediction.apply(lambda x: list(x.values()))

drop_feature.extend(['query_prediction', 'query_prediction_keys', 'query_prediction_values', 'max_query_prediction_keys'])

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[7]:


since = time.time()
# 分词, 对每一个item重复的词语去除重复
all_data['prefix_jieba'] = all_data.prefix.apply(lambda x: " ".join(jieba.cut(x, cut_all = False)))
all_data['prefix_jieba'] = all_data.prefix_jieba.apply(lambda x: " ".join(x.split()))

all_data['title_jieba'] = all_data.title.apply(lambda x: " ".join(jieba.cut(x, cut_all = False)))
all_data['title_jieba'] = all_data.title_jieba.apply(lambda x: " ".join(x.split()))

all_data['query_jieba'] = all_data.max_query_prediction_keys.apply(lambda x: " ".join(jieba.cut(x, cut_all = False)))
all_data['query_jieba'] = all_data.query_jieba.apply(lambda x: " ".join(x.split()))

drop_feature.extend(['prefix_jieba', 'title_jieba', 'query_jieba'])

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[8]:


since = time.time()
# 转换成拼音
p = Pinyin()
all_data['prefix_pinyin'] = all_data.prefix.apply(lambda x: p.get_pinyin(x, ' '))

drop_feature.append('prefix_pinyin')

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[9]:


since = time.time()
# 去掉prefix、title中的空格，转换大小写
all_data['prefix_fix'] = all_data.prefix.apply(lambda x: x.replace(' ', '').lower())
all_data['title_fix'] = all_data.title.apply(lambda x: x.replace(' ', '').lower())
all_data['query_fix'] = all_data.max_query_prediction_keys.apply(lambda x: x.replace(' ', '').lower())
all_data['query_prediction_keys_fix'] = all_data.query_prediction_keys.apply(lambda x: list(map(lambda item: item.replace(' ', '').lower(), x)))

drop_feature.extend(['prefix_fix', 'title_fix', 'query_fix', 'query_prediction_keys_fix'])

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[10]:


since = time.time()
# ----- length 特征 -----
list_length_feature = ['prefix', 'title', 'max_query_prediction_keys', 'query_prediction_values']

for feature in list_length_feature:
    printlog('计算' + feature + '长度', is_print_output)
    all_data[feature + '_length'] = all_data[feature].apply(lambda x: len(x))
for feature in ['prefix_jieba', 'title_jieba', 'query_jieba']:
    all_data[feature + '_length'] = all_data[feature].apply(lambda x: len(x.split()))

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[11]:


since = time.time()
# ----- nunique 特征 -----
list_nunique_feature = ['prefix', 'title', 'tag', 'max_query_prediction_keys', 'prefix_pinyin']

all_data['prefix_nunique_title'] = all_data.groupby('prefix').title.transform('nunique')
all_data['prefix_nunique_tag'] = all_data.groupby('prefix').tag.transform('nunique')

all_data['title_nunique_prefix'] = all_data.groupby('title').prefix.transform('nunique')
all_data['title_nunique_tag'] = all_data.groupby('title').tag.transform('nunique')
all_data['title_nunique_query'] = all_data.groupby('title').max_query_prediction_keys.transform('nunique')
all_data['title_nunique_prefix_pinyin'] = all_data.groupby('title').prefix_pinyin.transform('nunique')

all_data['tag_nunique_prefix'] = all_data.groupby('tag').prefix.transform('nunique')
all_data['tag_nunique_title'] = all_data.groupby('tag').title.transform('nunique')
all_data['tag_nunique_max_query'] = all_data.groupby('tag').max_query_prediction_keys.transform('nunique')

all_data['query_nunique_prefix'] = all_data.groupby('max_query_prediction_keys').prefix.transform('nunique')
all_data['query_nunique_title'] = all_data.groupby('max_query_prediction_keys').title.transform('nunique')
all_data['query_nunique_tag'] = all_data.groupby('max_query_prediction_keys').tag.transform('nunique')
all_data['query_nunique_prefix_pinyin'] = all_data.groupby('max_query_prediction_keys').prefix_pinyin.transform('nunique')

all_data['prefix_pinyin_nunique_prefix'] = all_data.groupby('prefix_pinyin').prefix.transform('nunique')
all_data['prefix_pinyin_nunique_title'] = all_data.groupby('prefix_pinyin').title.transform('nunique')
all_data['prefix_pinyin_nunique_tag'] = all_data.groupby('prefix_pinyin').tag.transform('nunique')
all_data['prefix_pinyin_nunique_query'] = all_data.groupby('prefix_pinyin').max_query_prediction_keys.transform('nunique')

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[12]:


since = time.time()
# is in feature
all_data['prefix_isin_title'] = all_data.apply(lambda row:1 if row['prefix_fix'] in row['title_fix'] else 0, axis = 1)
all_data['tag_isin_title'] = all_data.apply(lambda row:1 if row['tag'] in row['title_fix'] else 0, axis = 1)
all_data['query_isin_title'] = all_data.apply(lambda row:1 if row['query_fix'] in row['title_fix'] else 0, axis = 1)

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[13]:


similarity_func = [Levenshtein.ratio, Levenshtein.distance, lcsubstr_lens, lcseque_lens]
statistics_func = [max, min, np.mean, np.std]


# In[14]:


since = time.time()
# 计算prefix/title与query_prediction_keys相似度的list
list_with_query_prediction_keys_similarity = ['prefix_fix', 'title_fix']
for feature in list_with_query_prediction_keys_similarity:
    for func in similarity_func:
        printlog('计算' + feature + '与query_prediction_keys_' + func.__name__  + '相似度的list', is_print_output)
        all_data[feature + '_query_prediction_keys_' + func.__name__ +  '_list'] = all_data.apply(lambda row: [func(query, row[feature]) for query in row['query_prediction_keys_fix']], axis = 1)
        drop_feature.append(feature + '_query_prediction_keys_' + func.__name__ + '_list')
        
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[15]:


since = time.time()
# 计算prefix/title与query_prediction_keys相似度的list与query_prediction_values list的乘积list
list_with_query_prediction_keys_similarity_multiple = ['prefix_fix', 'title_fix']
multiple_similarity_func = [Levenshtein.ratio, Levenshtein.distance, lcsubstr_lens, lcseque_lens]
for feature in list_with_query_prediction_keys_similarity_multiple:
    for multiple_func in multiple_similarity_func:
        printlog('计算' + feature + '与query_prediction_values_' + multiple_func.__name__  + '相似度的list的乘积list', is_print_output)
        all_data[feature +  '_query_prediction_values_mutiple_' + multiple_func.__name__ + '_list'] = all_data.apply(lambda row: list(map(lambda x, y: x * y, row[feature + '_query_prediction_keys_' + multiple_func.__name__ +  '_list'], row['query_prediction_values'])), axis = 1)
        drop_feature.append(feature +  '_query_prediction_values_mutiple_' + multiple_func.__name__ + '_list')

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[16]:


since = time.time()
# 所有list相关统计的特征
# 找出所有list的特征
list_feature = list(filter(lambda x: x.find('list') != -1, drop_feature)) + ['query_prediction_values']
for feature in list_feature:
    for statistics in statistics_func:
        printlog('计算' + feature + '的' + statistics.__name__, is_print_output)
        all_data[feature + '_' + statistics.__name__] = all_data[feature].apply(lambda x: statistics(x) if x else np.nan)
        
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[17]:


since = time.time()
# 计算prefix/title/max_query_prediction_keys间的相似度
list_single_feature = ['prefix_fix', 'title_fix', 'query_fix']

for times in range(len(list_single_feature)):
    first_feature = list_single_feature.pop(0)
    for second_feature in list_single_feature:
        for func in similarity_func:
            printlog('计算' + first_feature + '与' + second_feature + '的' + func.__name__ + '相似度', is_print_output)
            all_data[func.__name__ + '_similarity_' + first_feature + '_with_' + second_feature] = all_data.apply(lambda row: func(row[first_feature], row[second_feature]), axis = 1)

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[18]:


since = time.time()
# 拼接回原数据
all_data = all_data.drop('query_prediction', axis = 1)
not_zip_all_data = pd.merge(not_zip_all_data, all_data, how = 'left', on = ['prefix', 'title', 'tag', 'diction_label'])

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[19]:


since = time.time()
# 算一些全局统计量
# ---- click 特征 ----
list_click_feature = ['prefix', 'title', 'tag', 'max_query_prediction_keys']

# 计算某特征单次点击
for feature in list_click_feature:
    printlog('计算' + feature + '点击次数', is_print_output)
    not_zip_all_data[feature + '_click'] = not_zip_all_data.groupby(feature)[feature].transform('count')
# 部分二元交叉点击
not_zip_all_data['prefix_title_click'] = not_zip_all_data.groupby(['prefix', 'title']).prefix.transform('count')
not_zip_all_data['prefix_tag_click'] = not_zip_all_data.groupby(['prefix', 'tag']).prefix.transform('count')
not_zip_all_data['title_tag_click'] = not_zip_all_data.groupby(['title', 'tag']).title.transform('count')
not_zip_all_data['title_max_query_prediction_keys_click'] = not_zip_all_data.groupby(['title', 'max_query_prediction_keys']).title.transform('count')
not_zip_all_data['tag_max_query_prediction_keys_click'] = not_zip_all_data.groupby(['tag', 'max_query_prediction_keys']).tag.transform('count')
# 部分三元交叉点击
not_zip_all_data['prefix_title_tag_click'] = not_zip_all_data.groupby(['prefix', 'title', 'tag']).prefix.transform('count')

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[20]:


since = time.time()
# 转换tag
encoder = preprocessing.LabelEncoder()
not_zip_all_data['tag'] = encoder.fit_transform(not_zip_all_data.tag)
encoder = preprocessing.LabelEncoder()
not_zip_all_data['prefix'] = encoder.fit_transform(not_zip_all_data.prefix)
encoder = preprocessing.LabelEncoder()
not_zip_all_data['title'] = encoder.fit_transform(not_zip_all_data.title)

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[21]:


# 保存数据
not_zip_all_data.drop(drop_feature, axis = 1).to_csv('Demo/Cases/base_1126.csv', index = False)


# In[22]:


time_elapsed = time.time() - all_start_time
print('final complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间

