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

# 数据查看和选择
# 1. 数据查看
print("查看数据")
# 查看数据
print(df.head(3))
print(df.tail(3))
print(df.sample(3)) # 随机3行
print()

print("基本信息")
# 基本信息
print(df.shape)
print(df.columns.tolist())
print(df.index)
print(df.dtypes)
print(df.info())
print(df.describe())

# 2. 数据选择
print("列选择和行选择")
# 列选择
print(df['A'])           # 选择单列
print(df[['A', 'C']])    # 选择多列

# 行选择
print(df[0:2])           # 切片选择行
print(df.iloc[0])        # 按位置选择行
print(df.loc[0])         # 按索引选择行

print("条件选择")
# 条件选择
print(df[df['A'] > 0])                    # 布尔索引
print(df[(df['A'] > 0) & (df['C'] < 2)]) # 多条件
print(df[df['E'].isin(['test'])])         # isin选择

print("iloc和loc")
# iloc和loc
print(df.iloc[0:2, 1:3])
print(df.loc[0:2, ['A', 'C']])

# 数据清洗和处理
# 1. 处理缺失值
df1 = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [10, 20, 30, 40]
})

print("检测缺失值")
# 检测缺失值
print(df1.isnull())
print(df1.isnull().sum())

# 处理缺失值
# 填充
df_filled = df1.fillna(0)
df_filled_mean = df1.fillna(df1.mean())
# 删除
df_dropped_row = df1.dropna()  # (axis = 0)
df_dropped_col = df1.dropna(axis = 1)

# 2. 数据转换
