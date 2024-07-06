import torch

# 构建可求导矩阵
x = torch.ones(3, 4, device='cuda', requires_grad=True)  # 法1，在构建的时候添加参数
print('Method one:', type(x))
print(x)

# 法2，手动设置
x = torch.ones(3, 4, device='cuda')
x.requires_grad = True
print('\nMethod two', type(x))
print(x)

b = torch.randn(3, 4,device='cuda', requires_grad=True)
t = x + b
y = t.sum()
print(type(t))
print(t)
print(type(y))
print(y)
