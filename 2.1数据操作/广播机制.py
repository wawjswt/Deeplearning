# 广播机制适用于不同形状的张量元素计算，基本原理是进行扩展或转换，使得张量形状相同
import torch

a = torch.arange(3).reshape((3, 1))  # 3行1列
b = torch.arange(2).reshape((1, 2))  # 1行2列
print(f'a：{a}\nb：{b}')
# 直接使用a+b会让a和b都扩展为三行两列的数组，a是复制列进行扩展，b是复制行进行扩展
a1 = torch.tensor([[0, 0], [1, 1], [2, 2]])
b1 = torch.tensor([[0, 1], [0, 1], [0, 1]])
print(f'a+b={a + b}')
print(f'a1+b1={a1 + b1}')
print(f'扩展计算结果', a1+b1==a+b)
