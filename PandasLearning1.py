# Python数据处理学习路径与建议(deepseek)
# Pandas
import numpy as np
import pandas as pd

# Pandas基础数据结构
# 1. Series
# 创建Series
series1 = pd.Series([1, 3, 5, np.nan, 6, 8])  # np.nan(Not a Number) 缺失值（无效）
series2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# 从字典创建
dictionary_data = {'a': 1, 'b': 2, 'c': 3}  # 自定义索引
series3 = pd.Series(dictionary_data)

print("Series")
# Series操作
print(series2.values)
print(series2.index)
print(series2['b'])
print(series2.head(2))
print()

# 2. DataFrame
# 创建DataFrame
df = pd.DataFrame({   #DataFrame: df
    'A': 1.0,
    'B': pd.Timestamp('20260101'),
    'C': pd.Series(1, index = list(range(4)), dtype = 'float32'),
    'D': np.array([3] * 4, dtype = 'int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'
})

print("从字典创建DataFrame")
# 从字典创建DataFrame
stockdata = {
    '股票代码': ['000001', '000002', '000003'],
    '价格': [25.6, 34.2, 18.9],
    '成交量': [1000000, 2000000, 1500000]
}
stock_df = pd.DataFrame(stockdata)

print(stock_df.shape)
print(stock_df.columns)
print(stock_df.index)
print(stock_df.dtypes)
print(stock_df.info())
print(stock_df.describe())
print()