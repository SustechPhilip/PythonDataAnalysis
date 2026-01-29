# Python数据处理学习路径与建议(deepseek)
# Pandas
import numpy as np
import pandas as pd
from numpy.ma.core import concatenate

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
df = pd.DataFrame({   #DataFrame: df DataFrame
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
df1 = pd.DataFrame({   # df1 DataFrame
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
# 类型转换
df1['A'] = df1['A'].astype('float32')

# 重命名列
df_renamed_column = df1.rename(columns = {'A': 'Alpha', 'B': 'Beta', 'C': 'Gamma'})

# 重置索引
df_reset = df1.reset_index()

# 替换值
df_replaced = df1.replace({1: 100, 5: 500})

# 应用函数
df1['A_squared'] = df1['A'].apply(lambda x: x**2)
df1['C_category'] = df1['C'].apply(lambda x: 'high' if x > 25 else 'low')

# 数据操作
# 1. 排序和排名
# 排序
df_sorted = df1.sort_values('A', ascending = False)
df_sorted_multi = df1.sort_values(['A', 'C'], ascending = [True, False])

# 排名
df1['A_rank'] = df1['A'].rank()

# 2. 分组聚合
# print('分组')
# 分组
# grouped = df.groupby('E')
# print(grouped.mean())
# print(grouped.agg({'A': 'mean', 'C': 'sum'}))
#
# 多级分组
# multi_grouped = df.groupby(['E', 'F'])
#
# 转换和过滤
# 转换
# df['group_mean'] = grouped['A'].transform('mean')
#
# 过滤
# filtered = grouped.filter(lambda x: x['A'].mean() > 1)

# 3. 数据合并
# 数据合并
# df_merged1 = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']})
# df_merged2 = pd.DataFrame({'A': ['A2', 'A3'], 'B': ['B2', 'B3']})
#
# 连接
# concatenated = pd.concat([df_merged1, df_merged2])
# concatenated_keys = pd.concat([df_merged1, df_merged2], keys = ['x', 'y'])

# 合并
# left = pd.DataFrame({'key': ['K0', 'K1'], 'A': ['A0', 'A1']})
# right = pd.DataFrame({'key': ['K0', 'K1'], 'B': ['B0', 'B1']})
# merged = pd.merge(left, right, on = 'key')

# 连接
# joined = ( left.set_index('key') ).join( right.set_index('key') )

# 时间序列处理
# 1. 时间索引
import pandas as pd
import numpy as np

print("\n创建时间序列")
# 创建时间序列
dates = pd.date_range('20230101', periods=6)
ts_df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print("原始数据:")
print(ts_df)
print()

# 按月切片 - 注意这里需要确保索引是DatetimeIndex类型
print("2023年1月的数据:")
try:
    # 方式1：使用部分日期字符串切片
    print(ts_df.loc['2023-01'])
except:
    # 方式2：如果上述不行，可以尝试这样
    print(ts_df[ts_df.index.month == 1])
print()

# 时间范围切片
print("2023-01-01到2023-01-03的数据:")
print(ts_df.loc['2023-01-01':'2023-01-03'])
print()

# 重采样示例
# 创建日频率数据
daily_data = pd.Series(np.random.randn(100),
                       index=pd.date_range('2023-01-01', periods=100, freq='D'))
print("日数据前5行:")
print(daily_data.head())
print()

# 按月重采样 - 计算每月平均值
# 注意：resample('M') 现在更推荐使用 'ME'（月末频率）
monthly_mean = daily_data.resample('ME').mean()
print("按月重采样后的月平均值:")
print(monthly_mean)
print()

# 其他有用的时间序列操作
print("其他时间序列操作示例:")
# 1. 按季度重采样
quarterly_mean = daily_data.resample('QE').mean()
print("季度平均值:", quarterly_mean.head())

# 时间偏移
print("\n时间偏移（7天后）:")
print(daily_data.head().index + pd.Timedelta(days=7))

# 提取时间组件
print("\n提取时间组件:")
daily_data_df = daily_data.to_frame('values')
daily_data_df['year'] = daily_data_df.index.year
daily_data_df['month'] = daily_data_df.index.month
daily_data_df['day'] = daily_data_df.index.day
daily_data_df['day_of_week'] = daily_data_df.index.dayofweek
print(daily_data_df.head())

# 2. 移动窗口操作

