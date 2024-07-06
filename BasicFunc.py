import torch

x = torch.empty(5, 3)  # 5x3 empty matrix
print(x)
# 返回tensor 张量，最基本的计算单元

y = torch.zeros(5, 3, dtype=torch.long)  # 5x3 zeros matrix
print(y)

z = torch.ones(5, 3)  # 5x3 ones matrix
print(z)

w = torch.rand(5, 3)  # 5x3 random matrix Value(0-1)
print(w)

x = torch.tensor([[5.3, 454, 343],
                  [122, 643,  36],
                  [11.3, 643, 15]])
# 手动创建tensor张量
print(x)
print(x.size())  # 打印对于张量结构，几行几列

# 矩阵加法1: c=a+b
print(y + z)  # 直接相加
# 矩阵加法2: c=torch.add(x,y)
print(torch.add(y, z))

# 索引,冒号表示取所有
print('索引,冒号表示取所有')
print(x[:, 1])  # 取第二列
print(x[:, 1:2])  # 取第二列
print(x[:, 1:3])  # 取第二列到第三列
print(x[:, 1:3:2])  # 取第二列到第三列，步长为2
print(x[:, 1:3:3])  # 取第二列到第三列，步长为3
print(x[1, :])  # 取第二行
print(x[1:3, ])  # 取第二行到第三行

# view改变矩阵维度,reshape
print('view改变矩阵维度')
print(x.view(9,1))  # 9x1
print(x.view(1,9))  # 1x9

# tensor和numpy计算必须进行转换
print('\ntensor和numpy计算必须进行转换')
y = x.numpy()
print(type(x))
print(type(y))
w = torch.from_numpy(y)
print(type(w))
# 两者之间可进行互相转换进行运算

# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype

if __name__ == "__main__":
    print("Starting")
    # 测试
