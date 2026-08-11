import torch

# 索引
a = torch.rand(4, 3, 28, 28)
print(a)

print('a[0]:', a[0])
print('a的size', a.size())
print('a[0]的size:', a[1].shape)
print('a[0,0]的size:', a[0, 0].size())
print('a[0,0,0]的size:', a[0, 0, 0].size())
print('a[0,0,0,0]的size:', a[0, 0, 0, 0].size())

b=torch.rand(3,4,5)
print('b',b)
print(b[:2]) # 读一个维度取0、1（不包括右括号）
print(b[:2,:3,:2]) # 取第一个维度（0，1）第二个维度（0，1，2），第三维（0，1）
print('b的第二个维度的1和3号',b.index_select(0,torch.tensor([0,2])))