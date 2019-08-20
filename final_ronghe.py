import pandas as pd
import numpy as np

# 读取两个子结果
test_wang = pd.read_csv('test_wang.csv', header=None)[0].values
test_dada = pd.read_csv('test_dada.csv', header=None)[0].values
# 融合
final_ronghe = test_wang * 0.48 + test_dada * 0.52
final_result = pd.DataFrame(final_ronghe, columns=['label'])
final_result['label'] = final_result['label'].apply(lambda x: 1 if x>0.388 else 0)
print('test转化个数：',final_result['label'].sum())
print('test shape：',final_result.shape)
# 提交最终结果
final_result['label'].to_csv("result.csv", index=False)