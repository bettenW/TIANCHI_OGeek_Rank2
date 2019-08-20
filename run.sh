#!/bin/sh
# 记录开始时间
start_tm=`date +%s%N`;

# 特征提取
python3 base_feature.py &
python3 convert_feature.py &
python3 count_vector_feature.py &
python3 base_add_feature.py &

wait

# lgb两个模型
python3 wanghe.py #特征提取与模型部分均放此文件完成
python3 lgb_online.py


# 记时间并输出
end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo $use_tm

# 结束
