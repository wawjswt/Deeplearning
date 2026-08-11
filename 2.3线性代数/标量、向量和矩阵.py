# 标量由只有一个元素的张量表示。 下面的代码将实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数。
import torch

x = torch.tensor(1)
y = torch.tensor(2)
print(f'x+y={x + y}{type(x + y)}\tx-y={x - y}\tx*y={x * y}\ty**x={y ^ x}')

# 向量可以被视为标量值组成的列表。 这些标量值被称为向量的元素（element）或分量（component）。 当向量表示数据集中的样本时，它们的值具有一定的现实意义。
