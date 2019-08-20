import sys
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import re
import gc
import jieba
import datetime  
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import f1_score
from sklearn import preprocessing
import difflib
import warnings
warnings.filterwarnings("ignore")
start = time.time()

# load data
train_df = pd.read_table('Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt', 
        names= ['prefix','query_prediction','title','tag','label'],quoting=3, header= None, encoding='utf-8').astype(str)
valid_df = pd.read_table('Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt', 
        names= ['prefix','query_prediction','title','tag','label'],quoting=3, header= None, encoding='utf-8').astype(str)
test_df = pd.read_table('Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt', 
        names= ['prefix','query_prediction','title','tag','label'], quoting=3,header= None, encoding='utf-8').astype(str)

train_df['trick'] = 1
valid_df['trick'] = 2
test_df['trick']  = 3
test_df['label'] = -1

train_df['label'] = train_df['label'].apply(lambda x: int(x))
valid_df['label'] = valid_df['label'].apply(lambda x: int(x))
test_df['label'] = test_df['label'].apply(lambda x: int(x))

data_df = pd.concat([train_df, valid_df, test_df],axis=0,ignore_index=True)
data = data_df.drop_duplicates(subset=['prefix','query_prediction','title','tag','trick'], keep='first').reset_index(drop = True)


# dict keys sort
def dict_sort(text):
    try:
        dicts = json.loads(text)
    except:
        dicts = {}
    return sorted(dicts.items(),key = lambda x:float(x[1]),reverse = True)
data['pred_list'] = data['query_prediction'].apply(dict_sort)
data['pred_len'] = data['pred_list'].apply(len)
data['prefix_len'] = data.prefix.apply(len)
data['title_len'] = data.title.apply(len)
data['is_prefix_in_title'] = data[['prefix','title']].apply(lambda row: row[1].find(row[0]),raw=True,axis=1)
data['title-prefix_len'] = data.title_len - data.prefix_len
data['ratio_len'] = data['prefix_len']/data['title_len']

def remove_cha(x):
    x = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', "", str(x))
    x = x.replace('2C', '')
    return x
def get_query_prediction_keys(x):
    try:
        x = json.loads(x)
    except:
        x = {}
    x = x.keys()
    x = [remove_cha(value) for value in x]    
    return ' '.join(x)
data['query_prediction_keys'] = data.query_prediction.apply(lambda x:get_query_prediction_keys(x))

def len_title_in_query(title, query):
    query = query.split(' ')
    if len(query) == 0:
        return 0
    l = 0
    for value in query:
        if value.find(title) >= 0:
            l += 1
    return l
data['is_title_in_query_keys'] = data.apply(lambda row:len_title_in_query(row['title'], row['query_prediction_keys']),axis = 1)
data['is_prefix_in_query_keys'] = data.apply(lambda row:len_title_in_query(row['prefix'], row['query_prediction_keys']),axis = 1)
data = data.drop(['query_prediction_keys'], axis=1)


# 参考https://github.com/luoling1993/TianChi_OGeek/blob/master/stat_engineering.py
def get_max_query_ratio(item):
    query_prediction = item['query_prediction']
    try:
        query_prediction = json.loads(query_prediction)
    except:
        query_prediction = {}
    title = item['title']
    if not query_prediction:
        return 0
    for query_wrod, ratio in query_prediction.items():
        if title == query_wrod:
            if float(ratio) > 0.1:
                return 1
    return 0

def get_word_length(item):
    item = str(item)
    word_cut = jieba.lcut(item)
    length = len(word_cut)
    return length

def get_small_query_num(item):
    small_query_num = 0
    try:
        item = json.loads(item)
    except:
        item = {}    
    for _, ratio in item.items():
        if float(ratio) <= 0.08:
            small_query_num += 1

    return small_query_num

data['max_query_ratio'] = data.apply(get_max_query_ratio, axis=1)
data['prefix_word_num'] = data['prefix'].apply(get_word_length)
data['title_word_num'] = data['title'].apply(get_word_length)
data['small_query_num'] =data['query_prediction'].apply(get_small_query_num)

# title 在 query_prediction 中构造
def is_in_query_loc(lst):
    try:
        pred = eval(lst[1])
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    for i,value in enumerate(pred):
        if(lst[0]==value[0]):
            return i
    return -1
data['title_in_query_loc'] = data[['title', 'query_prediction']].apply(is_in_query_loc, axis=1)

def is_in_query(lst):
    try:
        dicts = json.loads(lst[1])
    except:
        dicts = {}
    if lst[0] in dicts.keys():
        return 1
    else:
        return 0
data['title_in_query'] = data[['title', 'query_prediction']].apply(is_in_query, axis=1)

def _in_query_proba(lst):
    try:
        dicts = json.loads(lst[1])
    except:
        dicts = {}
    if lst[0] in dicts.keys():
        return dicts[lst[0]]
    else:
        return -1
data['title_in_query_proba'] = data[['title','query_prediction']].apply(_in_query_proba, axis=1)


# 参考https://github.com/GrinAndBear/OGeek/blob/master/create_feature.py
# 构造相似度列表
def extract_proba(pred):
    try:
        pred = eval(pred)
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    pred_proba_lst=[]
    for i in range(10):
        if len(pred)<i+2:
            pred_proba_lst.append(0)
        else:
            pred_proba_lst.append(float(pred[i][1]))
    return pred_proba_lst

def extract_prefix_pred_similarity(lst):
    try:
        pred = eval(lst[1])
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    prefix_pred_sim=[]
    for i in range(10):
        if len(pred)<i+2:
            prefix_pred_sim.append(0)
        else:
            prefix_pred_sim.append(difflib_similarity(lst[0],pred[i][0]))
    return prefix_pred_sim

def difflib_similarity(str1,str2):
    return difflib.SequenceMatcher(a=str1, b=str2).quick_ratio()

def extract_title_pred_similarity(lst):
    try:
        pred = eval(lst[1])
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    title_pred_sim=[]
    for i in range(10):
        if len(pred)<i+2:
            title_pred_sim.append(0)
        else:
            title_pred_sim.append(difflib_similarity(lst[0],pred[i][0]))
    return title_pred_sim

print('pred proba starting')
data['pred_proba_lst']=data['query_prediction'].apply(extract_proba)
print('prefix pred starting')
data['prefix_pred_sim']=data[['prefix','query_prediction']].apply(extract_prefix_pred_similarity,axis=1)
print('title pred starting')
data['title_pred_sim']=data[['title','query_prediction']].apply(extract_title_pred_similarity,axis=1)


# 对相似度列表进行统计
data['pred_proba_lst_max'] = data['pred_proba_lst'].apply(lambda x: max(x))
data['prefix_pred_sim_max'] = data['prefix_pred_sim'].apply(lambda x: max(x)) 
data['title_pred_sim_max'] = data['title_pred_sim'].apply(lambda x: max(x))
data['pred_proba_lst_std'] = data['pred_proba_lst'].apply(lambda x: np.std(x))
data['prefix_pred_sim_std'] = data['prefix_pred_sim'].apply(lambda x: np.std(x)) 
data['title_pred_sim_std'] = data['title_pred_sim'].apply(lambda x: np.std(x))
data['proba_max_prefix'] = data['prefix_pred_sim'].apply(lambda x: x[0])
data['proba_max_title'] = data['title_pred_sim'].apply(lambda x: x[0])
    
def do_mean(li):
    sums = 0
    for i in li:
        sums = sums + i
    return sums/len(li)
data['pred_proba_lst_mean'] = data['pred_proba_lst'].apply(do_mean)
data['prefix_pred_sim_mean'] = data['prefix_pred_sim'].apply(do_mean)
data['title_pred_sim_mean'] = data['title_pred_sim'].apply(do_mean)

data['prefix_title_sim']=data[['prefix','title']].apply(lambda row: difflib_similarity(row[0],row[1]),raw=True,axis=1)

def add_pred_similarity_feat(data):
    data['pred_proba_top5_sum'] = np.zeros(data.shape[0])
    data['pref_pred_sim_top5_sum'] = np.zeros(data.shape[0])
    data['title_pred_sim_top5_sum'] = np.zeros(data.shape[0])
    for i in range(5):
        data['pred_proba_top5_sum'] = data['pred_proba_top5_sum'] + data.pred_proba_lst.apply(lambda x:float(x[i]))
        data['pref_pred_sim_top5_sum'] = data['pref_pred_sim_top5_sum'] + data.prefix_pred_sim.apply(lambda x: float(x[i]))
        data['title_pred_sim_top5_sum'] = data['title_pred_sim_top5_sum'] + data.title_pred_sim.apply(lambda x: float(x[i]))
    data['pred_proba_top5_mean'] = data['pred_proba_top5_sum'].apply(lambda x: x/5)
    data['pref_pred_sim_top5_mean'] = data['pref_pred_sim_top5_sum'].apply(lambda x: x/5)
    data['title_pred_sim_top5_mean'] = data['title_pred_sim_top5_sum'].apply(lambda x: x/5)
    
    data['pred_proba_top3_sum'] = np.zeros(data.shape[0])
    data['pref_pred_sim_top3_sum'] = np.zeros(data.shape[0])
    data['title_pred_sim_top3_sum'] = np.zeros(data.shape[0])
    for i in range(3):
        data['pred_proba_top3_sum'] = data['pred_proba_top3_sum'] + data.pred_proba_lst.apply(lambda x:float(x[i]))
        data['pref_pred_sim_top3_sum'] = data['pref_pred_sim_top3_sum'] + data.prefix_pred_sim.apply(lambda x: float(x[i]))
        data['title_pred_sim_top3_sum'] = data['title_pred_sim_top3_sum'] + data.title_pred_sim.apply(lambda x: float(x[i]))
    data['pred_proba_top5_mean'] = data['pred_proba_top3_sum'].apply(lambda x: x/3)
    data['pref_pred_sim_top5_mean'] = data['pref_pred_sim_top3_sum'].apply(lambda x: x/3)
    data['title_pred_sim_top5_mean'] = data['title_pred_sim_top3_sum'].apply(lambda x: x/3)   
    return data
data = add_pred_similarity_feat(data)

# prefix和title的共现词
def word_match_share(row, item1, item2):
    item1_words = {}
    item2_words = {}
    for word in row[item1].split():
        item1_words[word] = 1
    for word in row[item2].split():
        item2_words[word] = 1
    if len(item1_words) == 0 or len(item2_words) == 0:
        return 0
    shared_words_in_item1 = [w for w in item1_words.keys() if w in item2_words]
    shared_words_in_item2 = [w for w in item2_words.keys() if w in item1_words]
    R = (len(shared_words_in_item1) + len(shared_words_in_item2))*1.0/(len(item1_words)+len(item2_words))
    return R
data['title_prefix_common_words'] = data.apply(lambda x: len(set(x['title'].split()).intersection(set(x['prefix'].split()))), axis = 1) # 有区分度

# 计算查询词prefix出现在title中的那个位置，前、后、中、没出现
def get_prefix_loc_in_title(prefix,title):
    if prefix not in title:
        return -1
    lens = len(prefix)
    if prefix == title[:lens]:
        return 0
    elif prefix == title[-lens:]:
        return 1
    else:
        return 2
data['prefix_loc'] = data.apply(lambda x : get_prefix_loc_in_title(x['prefix'],x['title']), axis=1) # 高区分度

# 压缩数据并提取 max_query_prediction_keys 特征
tag_encoder = preprocessing.LabelEncoder()
data_df['diction_label'] = tag_encoder.fit_transform(data_df.query_prediction)
zip_data = data_df.drop(['label', 'trick'], axis = 1).drop_duplicates().reset_index(drop = True)
def str_to_dict(dict_str):
    try:
        my_dict = eval(dict_str)
    except:
        my_dict = {}
    keys, values = my_dict.keys(), my_dict.values()
    my_dict = dict(zip(keys,list(map(lambda x: float(x), values))))
    return my_dict
zip_data['query_prediction'] = zip_data.query_prediction.apply(lambda x: str_to_dict(x))
zip_data['max_query_prediction_keys'] = zip_data.query_prediction.apply(lambda x: 'NoneDict' if x == {} else max(x, key = x.get))
zip_data = zip_data.drop('query_prediction', axis = 1)
data_df = pd.merge(data_df, zip_data, how = 'left', on = ['prefix', 'title', 'tag', 'diction_label'])
data_df.drop(['diction_label'], axis=1, inplace=True)


# 合并到总的数据集
data.drop(['label', 'pred_list', 'pred_proba_lst', 'prefix_pred_sim', 'title_pred_sim'], axis=1, inplace=True)
data_df = pd.merge(data_df, data, on=['prefix', 'query_prediction', 'title', 'tag', 'trick'], how='left')

strong_features = ['title_len', 'is_title_in_query_keys', 'title_word_num']
add_features = ['prefix_loc', 'title_in_query_loc']


# 构造ctr特征
print('N parts...')
train = data_df[:train_df.shape[0]][['tag','label']]
test_index = data_df[train_df.shape[0]:].index
n_parts = 5
index = []
for i in range(n_parts):
    index.append([])
tags = list(train['tag'].drop_duplicates().values)
for tag in tags:
    dt = train[train['tag']==tag]
    for k in range(2):
        lis = list(dt[dt['label']==k].sample(frac=1,random_state=2018).index)
        cut = [0]
        for i in range(n_parts):
            cut.append(int((i+1)*len(lis)/n_parts)+1)
        for j in range(n_parts):
            index[j].extend(lis[cut[j]:cut[j+1]])
se = pd.Series()
for r in range(n_parts):
    se = se.append(pd.Series(r+1,index=index[r]))
se = se.append(pd.Series(6,index=test_index)) 
data_df.insert(0,'n_parts',list(pd.Series(data_df.index).map(se).values))
# 进行平湖处理
#from smooth import HyperParam
from utility import HyperParam
label_feature = ['title', 'tag', 'prefix', 'max_query_prediction_keys'] + strong_features + add_features
feats = label_feature.copy()
label_feature.append('label')
label_feature.append('n_parts')
data = data_df[label_feature]
df_feature = pd.DataFrame()
data['cnt'] = 1
n_parts = 6
for feat in feats:
    feat_name = feat+'_ctr'
    print(feat_name)
    se = pd.Series()
    for i in range(n_parts):
        if i==0:
            df = data[data['n_parts'] == i+1][[feat]]
            temp = data[(data['n_parts'] != i + 1) & (data['n_parts'] <= 5)][[feat, 'label']].groupby(feat)['label'].agg({feat + '_click': 'sum', feat + '_count': 'count'})
            HP = HyperParam(1, 1)
            HP.update_from_data_by_moment(temp[feat + '_count'].values, temp[feat + '_click'].values)
            temp[feat + '_ctr_smooth'] = (temp[feat + '_click'] + HP.alpha) / (temp[feat + '_count'] + HP.alpha + HP.beta)
            se = se.append(pd.Series(df[feat].map(temp[feat + '_ctr_smooth']).values, index = df.index))
        elif i>=1 and i<=4:
            df = data[data['n_parts']==i+1][[feat]]
            temp = data[(data['n_parts']!=i+1)&(data['n_parts']<=5)&(data['n_parts']>=2)][[feat,'label']].groupby(feat)['label'].agg({feat + '_click': 'sum', feat + '_count': 'count'})
            HP = HyperParam(1, 1)
            HP.update_from_data_by_moment(temp[feat + '_count'].values, temp[feat + '_click'].values)
            temp[feat + '_ctr_smooth'] = (temp[feat + '_click'] + HP.alpha) / (
                        temp[feat + '_count'] + HP.alpha + HP.beta)
            se = se.append(pd.Series(df[feat].map(temp[feat + '_ctr_smooth']).values,index=df.index))
        elif i>=5:
            df = data[data['n_parts']==i+1][[feat]]
            temp = data[data['n_parts']<=5][[feat,'label']].groupby(feat)['label'].agg({feat + '_click': 'sum', feat + '_count': 'count'})
            HP = HyperParam(1, 1)
            HP.update_from_data_by_moment(temp[feat + '_count'].values, temp[feat + '_click'].values)
            temp[feat + '_ctr_smooth'] = (temp[feat + '_click'] + HP.alpha) / (
                    temp[feat + '_count'] + HP.alpha + HP.beta)
            se = se.append(pd.Series(df[feat].map(temp[feat + '_ctr_smooth']).values,index=df.index))
    df_feature[feat_name] = pd.Series(data.index).map(se)
for i in range(len(feats)):
    for j in range(len(feats)-i-1):
        feat_name = feats[i]+"_"+feats[i+j+1]+'_ctr'
        print(feat_name)
        se = pd.Series()
        for k in range(n_parts):
            if k==0:
                temp = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)].groupby([feats[i],feats[i+j+1]])['label'].agg({feat_name + '_click': 'sum', feat_name + '_count': 'count'})
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(temp[feat_name + '_count'].values, temp[feat_name + '_click'].values)
                temp[feat_name + '_ctr_smooth'] = (temp[feat_name + '_click'] + HP.alpha) / (
                        temp[feat_name + '_count'] + HP.alpha + HP.beta)
                dt = data[data['n_parts']==k+1][[feats[i],feats[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,temp[feat_name + '_ctr_smooth'].reset_index(),how='left',on=[feats[i],feats[i+j+1]])
                se = se.append(pd.Series(dt[feat_name + '_ctr_smooth'].values,index=list(dt['index'].values)))
            elif 1<=k and k<=4:
                temp = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)&(data['n_parts']>=2)].groupby([feats[i],feats[i+j+1]])['label'].agg({feat_name + '_click': 'sum', feat_name + '_count': 'count'})
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(temp[feat_name + '_count'].values, temp[feat_name + '_click'].values)
                temp[feat_name + '_ctr_smooth'] = (temp[feat_name + '_click'] + HP.alpha) / (
                        temp[feat_name + '_count'] + HP.alpha + HP.beta)
                dt = data[data['n_parts']==k+1][[feats[i],feats[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,temp[feat_name + '_ctr_smooth'].reset_index(),how='left',on=[feats[i],feats[i+j+1]])
                se = se.append(pd.Series(dt[feat_name + '_ctr_smooth'].values,index=list(dt['index'].values)))
            elif k>=5:
                temp = data[data['n_parts']<=5].groupby([feats[i],feats[i+j+1]])['label'].agg({feat_name + '_click': 'sum', feat_name + '_count': 'count'})
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(temp[feat_name + '_count'].values, temp[feat_name + '_click'].values)
                temp[feat_name + '_ctr_smooth'] = (temp[feat_name + '_click'] + HP.alpha) / (
                        temp[feat_name + '_count'] + HP.alpha + HP.beta)
                dt = data[data['n_parts']==k+1][[feats[i],feats[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,temp[feat_name + '_ctr_smooth'].reset_index(),how='left',on=[feats[i],feats[i+j+1]])
                se = se.append(pd.Series(dt[feat_name + '_ctr_smooth'].values,index=list(dt['index'].values)))
        df_feature[feat_name] = pd.Series(data.index).map(se)
data_df = pd.concat([data_df, df_feature], axis=1)
print('-------------making all sample ctr-----------------')
label_feature=['title', 'tag', 'prefix', 'query_prediction']
col_type = label_feature.copy()
label_feature.append('label')
label_feature.append('n_parts')
data_temp = data_df[label_feature]
df_feature = pd.DataFrame()
data_temp['cnt']=1
se = pd.Series()
for k in range(n_parts):
    if k==0:
        stat = data_temp[(data_temp['n_parts']!=k+1)&(data_temp['n_parts']<=5)].groupby(['prefix', 'query_prediction', 'title', 'tag'])['label'].mean()
        dt = data_temp[data_temp['n_parts']==k+1][['prefix', 'query_prediction', 'title', 'tag']]
        dt.insert(0,'index',list(dt.index))
        dt = pd.merge(dt,stat.reset_index(),how='left',on=['prefix', 'query_prediction', 'title', 'tag'])
        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
    elif 1<=k and k<=4:
        stat = data_temp[(data_temp['n_parts']!=k+1)&(data_temp['n_parts']<=5)&(data_temp['n_parts']>=2)].groupby(['prefix', 'query_prediction', 'title', 'tag'])['label'].mean()
        dt = data_temp[data_temp['n_parts']==k+1][['prefix', 'query_prediction', 'title', 'tag']]
        dt.insert(0,'index',list(dt.index))
        dt = pd.merge(dt,stat.reset_index(),how='left',on=['prefix', 'query_prediction', 'title', 'tag'])
        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
    elif k>=5:
        stat = data_temp[data_temp['n_parts']<=5].groupby(['prefix', 'query_prediction', 'title', 'tag'])['label'].mean()
        dt = data_temp[data_temp['n_parts']==k+1][['prefix', 'query_prediction', 'title', 'tag']]
        dt.insert(0,'index',list(dt.index))
        dt = pd.merge(dt,stat.reset_index(),how='left',on=['prefix', 'query_prediction', 'title', 'tag'])
        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
data_df['all_cvr'] = pd.Series(data_temp.index).map(se)
data_df.drop(['n_parts'], axis=1, inplace=True)

# add ratio feature
label_feature=['title', 'tag', 'prefix'] + strong_features + add_features
data_temp = data_df[label_feature]
df_feature = pd.DataFrame()
data_temp['cnt']=1
print('Begin ratio clcik...')
col_type = label_feature.copy()
n = len(col_type)
for i in range(n):
    col_name = "ratio_click_of_"+col_type[i]
    df_feature[col_name] =(data_temp[col_type[i]].map(data_temp[col_type[i]].value_counts())/len(data_df)*100).astype(int)
n = len(col_type)
for i in range(n):
    for j in range(n):
        if i!=j:
            col_name = "ratio_click_of_"+col_type[j]+"_in_"+col_type[i]
            se = data_temp.groupby([col_type[i],col_type[j]])['cnt'].sum()
            dt = data_temp[[col_type[i],col_type[j]]]
            cnt = data_temp[col_type[i]].map(data_df[col_type[i]].value_counts())
            df_feature[col_name] = ((pd.merge(dt,se.reset_index(),how='left',on=[col_type[i],col_type[j]]).sort_index()['cnt'].fillna(value=0)/cnt)*100).astype(int).values
data_df = pd.concat([data_df, df_feature], axis=1)
print('The end')

# add count feature
label_feature=['title', 'tag', 'prefix'] + strong_features
data_temp = data_df[label_feature]
df_feature = pd.DataFrame()
data_temp['cnt']=1
print('Begin count stat...')
col_type = label_feature.copy()
n = len(col_type)
for i in range(n):
    col_name = "cnt_of_"+col_type[i]
    se = (data_df[col_type[i]].map(data_df[col_type[i]].value_counts())).astype(int)
    semax = se.max()
    semin = se.min()
    df_feature[col_name] = ((se-se.min())/(se.max()-se.min())*100).astype(int).values
n = len(col_type)
for i in range(n):
    for j in range(n-i-1):
        col_name = "cnt_of_"+col_type[i+j+1]+"_and_"+col_type[i]
        se = data_temp.groupby([col_type[i],col_type[i+j+1]])['cnt'].sum()
        dt = data_temp[[col_type[i],col_type[i+j+1]]]
        se = (pd.merge(dt,se.reset_index(),how='left', on=[col_type[i],col_type[j+i+1]]).sort_index()['cnt'].fillna(value=0)).astype(int)
        semax = se.max()
        semin = se.min()
        df_feature[col_name] = ((se-se.min())/(se.max()-se.min())*100).fillna(value=0).astype(int).values
data_df = pd.concat([data_df, df_feature], axis=1)
print('The end')

# add nunique feature
label_feature=['title', 'tag', 'prefix'] + strong_features
data_temp = data_df[label_feature]
df_feature = pd.DataFrame()
data_temp['cnt']=1
print('Begin nunique stat...')
col_type = label_feature.copy()
n = len(col_type)
for i in range(n):
    for j in range(n):
        if i!=j:
            col_name = "count_type_"+col_type[j]+"_in_"+col_type[i]
            se = data_temp.groupby([col_type[i]])[col_type[j]].value_counts()
            se = pd.Series(1,index=se.index).sum(level=col_type[i])
            df_feature[col_name] = (data_temp[col_type[i]].map(se)).fillna(value=0).astype(int).values
data_df = pd.concat([data_df, df_feature], axis=1)
print('The end')

# CountVectorizer
data = data_df[['prefix', 'query_prediction', 'title', 'label']]
data.replace('nan',np.nan,inplace=True)
data['query_prediction'].fillna('{}',inplace=True)
data['title'].fillna('-1',inplace=True)
# prefix,title,query_prediction jieba分词
def get_cv_feature(dt):
    df = pd.DataFrame()
    for item in ['prefix', 'title']:
        print(item)
        stat = pd.DataFrame()
        stat[item] = dt[item].drop_duplicates().values
        stat[item+'_jieba'] = stat[item].apply(lambda x:' '.join(jieba.cut(str(x), cut_all=False)))
        df[item+'_jieba'] = pd.merge(dt,stat,how='left',on=item)[item+'_jieba']
    stat = pd.DataFrame()
    item = 'query_prediction'
    print(item)
    stat[item] = dt[item].drop_duplicates().values
    def getFeature(x):
        dct = json.loads(x)
        lst = []
        for k in dct.keys():
            lst.extend(jieba.cut(k,cut_all=False))
        return ' '.join(lst)
    stat['query_prediction_jieba'] = stat['query_prediction'].apply(getFeature)
    df[item+'_jieba'] = pd.merge(dt,stat,how='left',on=item)[item+'_jieba']
    return df
df = get_cv_feature(data)
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
cntv=CountVectorizer()
data['label'] = data['label'].astype(int)
vector_feature = ['prefix_jieba','query_prediction_jieba','title_jieba']
train_index = data[data['label']>=0].index.tolist()
test_index = data[data['label']==-1].index.tolist()
train_sp = pd.DataFrame()
test_sp = pd.DataFrame()
for feature in vector_feature:
    print(feature)
    cntv.fit(df[feature])
    train_sp = sparse.hstack((train_sp,cntv.transform(df.loc[train_index][feature]))).tocsr()
    test_sp = sparse.hstack((test_sp,cntv.transform(df.loc[test_index][feature]))).tocsr()
print(train_sp.shape)
print(test_sp.shape)

#######################################################
# labelencoder
col_encoder = LabelEncoder()
for feat in ['prefix', 'title', 'tag', 'query_prediction', 'max_query_prediction_keys']:
    col_encoder.fit(data_df[feat])
    data_df[feat] = col_encoder.transform(data_df[feat])
data_df.drop(['query_prediction', 'title_title_len_ctr', 'prefix', 'title'], axis=1, inplace=True)

# 切分数据
y_train = data_df['label'][:-200000].values
X_train = sparse.hstack((data_df.drop(['label'], axis=1)[:-200000].astype(float),train_sp)).tocsr()
X_test = sparse.hstack((data_df.drop(['label'], axis=1)[-200000:].astype(float),test_sp)).tocsr()
del data_df
gc.collect()

# 合并prefix_contains_tag
pct = sparse.load_npz('Demo/Cases/prefix_contains_tag.npz')
X_train = sparse.hstack((X_train,pct[:-200000])).tocsr()
X_test = sparse.hstack((X_test,pct[-200000:])).tocsr()

print(X_train.shape)
print(X_test.shape)

dt = time.time()-start
print('数据准备时间: ',dt,'s')

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 144,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':3,
    'reg_lambda':5
}
print('=======================  train+valid训练  =========iter987==========')
lgb_train = lgb.Dataset(X_train, y_train)
gbm = lgb.train(params, lgb_train, valid_sets=[lgb_train], num_boost_round = 987, verbose_eval=50)
print('模型训练时间: ',time.time()-dt-start,'s')

# 预测结果并保存
test_df['label'] = gbm.predict(X_test, num_iteration=987)
test_df['label'].to_csv('test_wang.csv', index=False)