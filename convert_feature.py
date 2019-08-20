
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
import jieba, os, Levenshtein, time
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from utility import *
from sklearn import preprocessing
import numpy as np 
import random, copy
from xpinyin import Pinyin


# In[2]:
print('run convert_feature')

# 配置信息
is_print_output = True
frac_size = 0.5
is_fill_na = False
is_online = False
# 转换率折数
sec_size = 5
print(frac_size)
all_start_time = time.time()


# In[3]:


since = time.time()
# 读入数据
train_data = read_file('Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt')
val_data = read_file('Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt')
test_data = read_file('Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt', True)

if not is_online:
    val_data['label'] = -1
    print('当前跑的是线下！')
    
# 拼接数据一起做特征
all_data = pd.concat((train_data, val_data, test_data), axis = 0, ignore_index = True, sort = False)

# 修正query_prediction为空
all_data.loc[all_data.query_prediction == '', 'query_prediction'] = '{}'
# 转换label的数据类型
all_data['label'] = all_data.label.astype('int')
# all_data['prefix'] = all_data.prefix.apply(lambda x: x.replace(' ', '').lower())

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


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
zip_data['max_query_prediction_keys'] = zip_data.query_prediction.apply(lambda x: 'NoneDict' if x == {} else max(x, key = x.get))

p = Pinyin()
zip_data['prefix_pinyin'] = zip_data.prefix.apply(lambda x: p.get_pinyin(x, ' '))

# # 去掉prefix中的空格和转换大小写
zip_data['prefix_fix'] = zip_data.prefix.apply(lambda x: x.replace(' ', '').lower())

zip_data = zip_data.drop('query_prediction', axis = 1)
# print(set(zip_data.columns)&set(all_data.columns))
all_data = pd.merge(all_data, zip_data, how = 'left', on = ['prefix', 'title', 'tag', 'diction_label'])

# drop_feature = list(all_data.columns)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[5]:


since = time.time()
# 分区算转换率
all_data['old_index'] = all_data.index
drop_feature = list(all_data.columns)

# 初始化一次随机划分
random.seed(19960121)
all_data['random_sector'] = [random.randint(1, sec_size) for num in range(len(all_data))]
all_data.loc[all_data.label == -1, 'random_sector'] = 0

convert_feature = ['prefix', 'title', 'tag', 'max_query_prediction_keys', 'prefix_pinyin', 'prefix_fix']

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[6]:


since = time.time()

for index, feature in enumerate(convert_feature):
    printlog('计算' + feature + '转换率', is_print_output)
    for sec in range(sec_size + 1):
        temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)][[feature, 'label']]
        
        if sec != 0:
            temp = temp.sample(frac = frac_size, random_state = 19960121).reset_index(drop = True)
        
        temp[feature + '_all_count'] = temp.groupby(feature).label.transform('count')
        temp[feature + '_label_count'] = temp.groupby(feature).label.transform('sum')
        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(temp[feature + '_all_count'].values, temp[feature + '_label_count'].values)
        temp[feature + '_convert'] = (temp[feature + '_label_count'] + HP.alpha) / (temp[feature + '_all_count'] + HP.alpha + HP.beta)
        temp = temp[[feature, feature + '_convert']].drop_duplicates()
        
        sec_data = copy.deepcopy(all_data[all_data.random_sector == sec])
        sec_data = pd.merge(sec_data, temp, on = [feature], how = 'left')
        if is_fill_na:
            sec_data[feature + '_convert'].fillna(HP.alpha / (HP.alpha + HP.beta), inplace = True)
        if sec:
            new_all_data = pd.concat((new_all_data, sec_data))
        else:
            new_all_data = copy.deepcopy(sec_data)
    all_data = copy.deepcopy(new_all_data)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[7]:


since = time.time()

for index, feature_temp in enumerate([['prefix', 'title'], ['prefix', 'tag'], ['title', 'tag']]):
    first_feature, second_feature = feature_temp
    printlog('计算' + first_feature + '和' + second_feature + '转换率', is_print_output)
    
    for sec in range(sec_size + 1):
        temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)][[first_feature, second_feature, 'label']]
#         temp[first_feature + '_' + second_feature + '_convert'] = temp.groupby([first_feature, second_feature]).label.transform('count') / temp.groupby([first_feature, second_feature]).label.transform('sum')
        if sec != 0:
            temp = temp.sample(frac = frac_size, random_state = 19960121).reset_index(drop = True)
        
        temp[first_feature + '_' + second_feature + '_all_count'] = temp.groupby([first_feature, second_feature]).label.transform('count')
        temp[first_feature + '_' + second_feature + '_label_count'] = temp.groupby([first_feature, second_feature]).label.transform('sum')
        
        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(temp[first_feature + '_' + second_feature + '_all_count'].values, temp[first_feature + '_' + second_feature + '_label_count'].values)
        temp[first_feature + '_' + second_feature + '_convert'] = (temp[first_feature + '_' + second_feature + '_label_count'] + HP.alpha) / (temp[first_feature + '_' + second_feature + '_all_count'] + HP.alpha + HP.beta)
        
        temp = temp[[first_feature, second_feature, first_feature + '_' + second_feature + '_convert']].drop_duplicates()
        
        sec_data = copy.deepcopy(all_data[all_data.random_sector == sec])
        sec_data = pd.merge(sec_data, temp, on = [first_feature, second_feature], how = 'left')
        if is_fill_na:
            sec_data[first_feature + '_' + second_feature + '_convert'].fillna(HP.alpha / (HP.alpha + HP.beta), inplace = True)
        if sec:
            new_all_data = pd.concat((new_all_data, sec_data))
        else:
            new_all_data = copy.deepcopy(sec_data)
    all_data = copy.deepcopy(new_all_data)
    
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[8]:


since = time.time()
for sec in range(sec_size + 1):
    temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)][['prefix', 'title', 'tag', 'label']]
#     temp['prefix_title_tag_convert'] = temp.groupby(['prefix', 'title', 'tag']).label.transform('count') / temp.groupby(['prefix', 'title', 'tag']).label.transform('sum')
    if sec != 0:
        temp = temp.sample(frac = frac_size, random_state = 19960121).reset_index(drop = True)
    
    temp['prefix_title_tag_all_count'] = temp.groupby(['prefix', 'title', 'tag']).label.transform('count')
    temp['prefix_title_tag_label_count'] = temp.groupby(['prefix', 'title', 'tag']).label.transform('sum')
    
    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(temp['prefix_title_tag_all_count'].values, temp['prefix_title_tag_label_count'].values)
        
    temp['prefix_title_tag_convert'] = (temp['prefix_title_tag_label_count'] + HP.alpha) / (temp['prefix_title_tag_all_count'] + HP.alpha + HP.beta)
    
    temp = temp[['prefix', 'title', 'tag', 'prefix_title_tag_convert']].drop_duplicates()
    
    sec_data = copy.deepcopy(all_data[all_data.random_sector == sec])
    sec_data = pd.merge(sec_data, temp, on = ['prefix', 'title', 'tag'], how = 'left')
    if is_fill_na:
        sec_data['prefix_title_tag_convert'].fillna(HP.alpha / (HP.alpha + HP.beta), inplace = True)
    if sec:
        new_all_data = pd.concat((new_all_data, sec_data))
    else:
        new_all_data = copy.deepcopy(sec_data)
all_data = copy.deepcopy(new_all_data)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[11]:


all_data.sort_values(by = 'old_index', inplace = True)


# In[12]:


all_data.drop(drop_feature, axis = 1).to_csv('Demo/Cases/ultron4_convert_' + str(sec_size) + '_' + str(frac_size).replace('.', '_') + '_' + str(is_online) + '.csv', index = False)


# In[13]:


time_elapsed = time.time() - all_start_time
print('final complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[14]:


all_data.random_sector.value_counts()

