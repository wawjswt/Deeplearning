# 检查cuda和cudnn是否安装成功的代码
import torch
# 若正常则静默
a = torch.tensor(1.)
# 若正常则静默
print(a.cuda())
# 若正常则返回 tensor(1., device='cuda:0')
from torch.backends import cudnn
# 若正常则静默
print(cudnn.is_available())
# 若正常则返回 True
print(cudnn.is_acceptable(a.cuda()))
# 若正常则返回 True
print(torch.cuda.is_available())
# 返回GPU的数量
print(torch.cuda.device_count())

print(f'pytorch的版本是:',torch.__version__)

if torch.cuda.is_available():
    print(f"CUDA is available. CUDA Version: {torch.version.cuda}")
else:
    print("CUDA is not available.")