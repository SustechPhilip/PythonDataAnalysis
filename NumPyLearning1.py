# Python数据处理学习路径与建议(deepseek)
# NumPy
import numpy as np

# NumPy基础部分
# 1. 创建数组
# 从列表创建
array1 = np.array([1, 2, 3, 4, 5 ])
array2 = np.array([[1, 2, 3 ], [4, 5, 6]]) # 2dim数组
print(array1, array2)

# 特殊数组
zeroArray = np.zeros((3, 3)) # 全0数组
oneArray = np.ones((3, 3)) # 全1数组
emptyArray = np.empty((3, 3)) # 未初始化数组
fullArray = np.full((3, 3), 1) # 全特定值矩阵
identityArray = np.eye(3) # 单位矩阵

# 序列数组
rangeArray = np.arange(0, 10, 2) # [0, 2, 4, 6, 8] (start, stop), step)
linspaceArray = np.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1] (start, stop], num)

# 随机数组
randomArray = np.random.rand(3, 3) # 均匀随机分布
normalRandomArray = np.random.randn(3, 3) # 正态随机分布
randomIntegerArray = np.random.randint(0, 10, (3, 3)) # 整数随机分布


# 2. 数组属性
print("dimension", array2.ndim)
print("shape", array2.shape)
print("size", array2.size)
print("datatype", array2.dtype)
print("itemsize", array2.itemsize)


# NumPy数组操作
# 1. 索引和切片
array3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(array3)

print("基本索引")
# 基本索引
print(array3[0, 0])
print(array3[1])

print("切片")
# 切片
print(array3[0:2, 1:3])
print(array3[:, 1:3])
print(array3[0:2, :])

print("布尔索引")
# 布尔索引
bool_idx1 = array3 > 5
bool_idx2 = array3 > 10
print(array3[bool_idx1])
print(array3[bool_idx2])

print("花式索引")
# 花式索引
print(array3[[0, 2], :])
print(array3[:, [1, 3]])

# 2. 数组变形
array4 = np.arange(12)
print(array4)
print("改变形状")
# 改变形状
reshaped = array4.reshape(3, 4) # (row, column)
print(reshaped)
flattened = array4.flatten()
print(flattened)
raveled = array4. ravel()
print(raveled)

print("转置")
# 转置
transposed = reshaped.T
print(transposed)

print("增加维度")
# 增加维度
expanded1 = np.expand_dims(array4, axis = 0)
print(expanded1)
expanded2 = np.expand_dims(array4, axis = -1)
print(expanded2)
squeezed = np.squeeze(expanded1)
print(squeezed)

# 3. 数组拼接和分割
array5 = np.array([[1, 2], [3, 4]])
array6 = np.array([[5, 6], [7, 8]])
print("拼接")
# 拼接
vstack = np.vstack((array5, array6))
print(vstack)
hstack = np.hstack((array5, array6))
print(hstack)
concatenated = np.concatenate((array5, array6), axis = 0)
print(concatenated)

print("分割")
# 分割
array7 = np.arange(12).reshape(3, 4)
print(array7)
split_array7 = np.split(array7, 2, axis = 1)
print(split_array7)
vsplit_array7 = np.vsplit(array7, 3)
print(vsplit_array7)


# NumPy数学运算
# 1. 基本运算
array8 = np.array([1, 2, 3])
array9 = np.array([4, 5, 6])

print("元素级运算")
# 元素级运算
print(array8 + array9)
print(array8 - array9)
print(array8 * array9)
print(array8 / array9)
print(array8 ** 2)

print("矩阵乘法")
# 矩阵乘法
matrix_a = np.array([[1, 2], [3, 4]])
print(matrix_a)
matrix_b = np.array([[5, 6], [7, 8]])
print(matrix_b)
dot_product = np.dot(matrix_a, matrix_b)
print(dot_product)

# 2. 通用函数
array10 = np.array([1, 4, 9, 16])

print("数学函数")
# 数学函数
print(np.sqrt(array10))
print(np.exp(array10))
print(np.log(array10))
print(np.sin(array10))
print(np.cos(array10))

print("统计数字特征")
# 统计数字特征
print(np.sum(array10))
print(np.mean(array10))
print(np.std(array10))
print(np.var(array10))
print(np.min(array10))
print(np.max(array10))
print(np.argmax(array10))

print("沿轴计算")
# 沿轴计算
array_2d = np.array([[1, 2], [3, 4]])
print(np.sum(array_2d, axis = 0))  # column
print(np.sum(array_2d, axis = 1))  # row

# 3. 广播机制
print("标量与数组")
# 标量与数组
array11 = np.array([1, 2, 3])
result1 = array11 + 11
print(result1)  # [12, 13, 14]

print("不同形状数组")  # 兼容的形状 #
# 不同形状数组
array12 = np.array([[1], [2], [3]])  #  [[1, 1, 1] [2, 2, 2] [3, 3, 3]]
array13 = np.array ([4, 5, 6])  #  [[4, 5, 6] [4, 5, 6] [4, 5, 6]]
result2 = array12 + array13
print(result2)  # [[5 6 7] [6 7 8] [7 8 9]]

print("归一化")
# 归一化（例子）
# 创建随机数据 (5个样本，3个特征)
np.random.seed(42)
data = np.random.rand(5, 3) * 100
print("原始数据:\n", data)
print("形状:", data.shape)  # (5, 3)
print()

# 计算每列的均值和标准差
mean_per_column = data.mean(axis=0)  # shape: (3,)
std_per_column = data.std(axis=0)    # shape: (3,)

print("每列均值:", mean_per_column)  # (3, ) 形状 (1, 3) 广播 (5, 3)
print("每列标准差:", std_per_column)  # (3, )
print()

# 广播进行归一化
normalized = (data - data.mean(axis=0)) / data.std(axis=0)  # 归一化（广播机制）

print("归一化后的数据:\n", normalized)
print("\n验证归一化效果:")
print("归一化后每列均值:", normalized.mean(axis=0))  # 应该接近0
print("归一化后每列标准差:", normalized.std(axis=0))  # 应该为1




