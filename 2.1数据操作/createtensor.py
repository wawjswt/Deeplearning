# pytorch学习
from __future__ import print_function  # 规范化print()函数，在python2中，print函数不需要括号，这几句话让print()函数必须使用括号
import torch
import numpy as np

# import from numpy
a = np.array([2, 3.3])
print(a)
b = torch.from_numpy(a)
print(b)

# import from list
a = [2, 3.3]
print(a)
b = torch.tensor(a)
print(b)
c=torch.FloatTensor(2,2,2)
print(c)

# uninitialized
d = torch.empty(5, 3)
print(d)
e = torch.empty(5, 3, dtype=torch.long)
print(e)
f = torch.empty(5, 3, dtype=torch.float64)
print(f)

m=torch.full([2,3,4], 7)
print(m)

n=torch.eye(3,3)
print(n)