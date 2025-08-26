# 使用索引进行访问元素
import torch
x=torch.arange(12)
x=torch.reshape(x,(3,4))
print(x)
print(x[-1])  # 最后一行
print(x[-1,-1])  #最后一个元素
x[-1,-1]=7758258
print(f'修改最后一个元素后的x：{x}')

# id()函数用来指出变量的地址，多用x+=y，而非x=x+y来减少内存使用，前者x计算前后地址相同，后者则地址不同，占用更多的内存
print(id(x))