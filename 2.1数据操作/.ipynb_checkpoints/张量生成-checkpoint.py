# 为了能够完成各种数据操作，我们需要某种方法来存储和操作数据。
# 通常，我们需要做两件重要的事：（1）获取数据；（2）将数据读入计算机后对其进行处理。 如果没有某种方法来存储数据，那么获取数据是没有意义的。

import torch

x = torch.arange(12)
print(x)
print('x的形状:', x.shape)
print('x的大小（多少个元素）:', x.numel())
X = x.reshape(3, 4)
print(f'改变形状:{X}')
print('X的形状:', X.shape)
# 使用-1可以实现自动计算该维度的长度

# 全零、全1张量、正态、自给列表
print("全零：", torch.zeros([2, 3, 4]))
print("全1：", torch.ones([2, 3, 4]))
print("正态：", torch.randn([2, 3, 4]))
list1 = [[[1, 1, 4], [5, 1, 4]], [[7, 7, 5], [2, 5, 8]]]
print("列表：", torch.tensor(list1))
