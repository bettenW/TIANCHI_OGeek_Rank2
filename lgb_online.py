
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import sparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn import metrics
import os, time, datetime
import numpy as np
from sklearn import preprocessing
from utility import read_file, lcseque_lens, lcsubstr_lens


# In[2]:


# 配置信息


# In[3]:


since = time.time()
all_data = pd.read_csv('Demo/Cases/base_1126.csv')
add_convert = pd.read_csv('Demo/Cases/ultron4_convert_5_0_5_False.csv')
add_convert = add_convert.drop(['prefix_convert', 'title_convert', 'tag_convert'], axis = 1)
all_data = pd.concat((all_data, add_convert), axis = 1)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[ ]:


since = time.time()
all_data = all_data.drop(['prefix', 'title', 'random_sector'], axis = 1)

test_data = all_data[all_data.label == -1].drop('label', axis = 1).reset_index(drop = True)
train_data = all_data[all_data.label != -1].reset_index(drop = True)
labels = train_data.pop('label')

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[ ]:


since = time.time()
all_cv = sparse.load_npz("Demo/Cases/all_cv2.npz").tocsr()
add_cv = sparse.load_npz("Demo/Cases/prefix_contains_tag.npz").tocsr()
all_cv = sparse.hstack((all_cv, add_cv)).tocsr()
# all_cv = sparse.hstack((all_cv, add_cv)).tocsr()

train_cv = all_cv[:-200000]
test_cv = all_cv[-200000:]
train_data = sparse.hstack((train_data, train_cv)).tocsr()
test_data = sparse.hstack((test_data, test_cv)).tocsr()
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间


# In[7]:


clf = lgb.LGBMClassifier(
        boosting_type = 'gbdt', num_leaves = 64, reg_alpha = 5, reg_lambda = 5,
        n_estimators = 4053, objective = 'binary',
        subsample = 0.7, colsample_bytree = 0.7, subsample_freq = 1,
        learning_rate = 0.05, random_state = 8012, n_jobs = -1)
    
clf.fit(train_data, labels, eval_set = [(train_data, labels)], verbose = 50)


# In[ ]:


test_result = clf.predict_proba(test_data)
test_data = all_data[all_data.label == -1].drop('label', axis = 1).reset_index(drop = True)
test_data['label'] = test_result[:, 1]
test_data['label'] = test_data.label.apply(lambda x:1 if x >= 0.36 else 0)
# test_data[['label']].to_csv(datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv", index = None,header = None)


# In[ ]:


test_res = pd.DataFrame(test_result[:, 1], columns=['label'])
# test_res['label'].to_csv('test_dada.csv', index=False)

# 读取第一个模型的结果
test_wang = pd.read_csv('test_wang.csv', header=None)[0].values
test_dada = test_res['label'].values
# 融合
final_ronghe = test_wang * 0.48 + test_dada * 0.52
final_result = pd.DataFrame(final_ronghe, columns=['label'])
final_result['label'] = final_result['label'].apply(lambda x: 1 if x>0.388 else 0)
print('test转化个数：',final_result['label'].sum())
print('test shape：',final_result.shape)
# 提交最终结果
final_result['label'].to_csv("result.csv", index=False)