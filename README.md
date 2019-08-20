# TIANCHI_OGeek_Rank2
# TIANCHI天池-OGeek算法挑战赛（亚军）

### 1.解题思路

- 模型数量:2个lightgbm模型的简单加权融合, 最终结果由两套代码生成结果加权融合所得, 两个结果均有LightGBM训练所得, 不同的是特征的差异性. 
- 参数的差异性。
- 主要特征:
-- a.基础特征:
       1).主要包括prefix/title/max_query_prediction_keys相似度的系列特征, 对于一系列的统计list的数学统计特征, 分别统计list中的最大/最小
    均值/标准差/众数等等;
       2).转换率特征, 关于prefix/title/tag的相关转换率以及组合的一些转换率, 统计的时候用了一些简单的sample trick, 防止过拟合;
       3).点击特征, 统计关于prefix/title/tag的单个点击以及组合的点击特征;
       4).统计nunique特征, 例如prefix下有多少个title等等类似的特征;
-- b.countvector特征:对于prefix/title/query_prediction用cv变换成词的稀疏矩阵输入进模型一起训练, 另外还有prefix下包括哪些tag字符串的countvector;
### 2.哪部分是开源的
  在utility文件中, 有一些功能函数, 例如计算两个字符串间的各种相似度, 以及转换率平滑函数HyperParam, 都是开源的
### 3.额外需要安装的包
  lightgbm, jieba, difflib(python3自带), Levenshtein(pip install --user python-Levenshtein), sklearn, scipy, tqdm, xpinyin
### 4.运行时长
  3个小时
### 5.提及结果文件及位置
  /home/admin/jupyter目录下的result.csv文件
### 6.无停用词语
### 7.队伍名称:去网吧里偷耳机
  队长简介:王贺 武汉大学硕士毕业 京东算法工程师  
  鱼遇雨欲语与余
### 8.执行方式:执行 ./run.sh
