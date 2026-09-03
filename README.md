# Deeplearning

一个用于整理深度学习学习笔记、PyTorch 基础代码和小型实验的仓库。内容按照“数学与数据基础 → 自动求导 → 模型训练 → 综合实验”逐步展开，适合配合 Jupyter Notebook、Python 脚本和练习题学习。

本仓库的核心目标不是提供一个可直接部署的完整应用，而是通过可运行的最小示例理解深度学习的基本组成：张量、矩阵运算、数据预处理、计算图、损失函数、梯度下降和模型训练。

## 学习路线

建议按以下顺序学习：

1. 先学习 `2.1数据操作`，掌握张量创建、索引、切片、广播和维度变换；
2. 学习 `2.2数据预处理`，理解数据读取、缺失值处理、特征构造和数据集划分；
3. 学习 `2.3线性代数` 和 `2.4矩阵计算`，建立向量、矩阵和矩阵求导基础；
4. 学习 `2.5自动求导`，理解计算图、梯度和 `backward()`；
5. 学习 `2.6 概率`，补充概率相关知识；
6. 进入 `3.1 线性回归`，把数据、模型、损失、反向传播和参数更新串成完整训练流程；
7. 最后阅读 `convolution`、`EmotionClassification`、`TransferingLearning` 和 `picturestyle` 中的综合实验。

## 内容概览

### 基础知识

| 目录 | 内容 |
| --- | --- |
| `2.1数据操作` | 张量创建、索引、维度变换、广播和常用运算 |
| `2.2数据预处理` | 数据读取、缺失值处理、特征处理和数据集划分 |
| `2.3线性代数` | 标量、向量、矩阵及常见矩阵运算 |
| `2.4矩阵计算` | 矩阵运算与求导 |
| `2.5自动求导` | PyTorch 计算图、梯度和 `backward()`，附知识点与训练题 |
| `2.6 概率` | 概率相关学习笔记 |
| `3.1 线性回归` | 线性回归、平方损失、梯度下降、优化方法和完整训练流程；包含详细整理文档与 30 道互动习题 |

### 综合实验

- `convolution`：使用卷积神经网络进行 MNIST 手写数字识别。
- `EmotionClassification`：基于中文分词和词袋模型的文本情感分类。
- `TransferingLearning`：使用预训练 ResNet 进行蚂蚁/蜜蜂图像分类的迁移学习实验。
- `picturestyle`：图像风格迁移实验。

### 数据与脚本

- `data/house_tiny.csv`：数据预处理示例使用的小型 CSV 数据。
- `dataset/`：根目录下的蚂蚁/蜜蜂图像数据，按 `train`、`val` 和类别目录组织。
- `TransferingLearning/data/`：迁移学习脚本实际使用的蚂蚁/蜜蜂图像数据。
- `EmotionClassification/data/`：文本情感分类实验使用的正面、负面文本和模型文件。
- 根目录下的 `BasicFunc.py`、`autograd.py`、`read_data.py`、`维度变换.py`、`测试cuda版本.py` 等文件：独立的基础示例和环境检查脚本。

## 3.1 线性回归专题

`3.1 线性回归` 是从“会写 PyTorch 张量代码”过渡到“理解模型训练”的关键章节。

### 学习资料

- [线性回归从零开始 Notebook](./3.1%20线性回归/线性回归从零开始.ipynb)：手动实现数据生成、模型、平方损失、小批量迭代器、自动求导和 SGD。
- [线性回归简洁实现 Notebook](./3.1%20线性回归/线性回归简洁实现.ipynb)：使用 `nn.Linear`、`MSELoss`、`DataLoader` 和优化器完成同一个任务。
- [优化方法基础 Notebook](./3.1%20线性回归/优化方法基础.ipynb)：理解梯度下降、学习率和小批量梯度的基本思想。
- [线性回归完整整理](./3.1%20线性回归/线性回归完整整理.md)：包含公式、代码逐段解释、两种实现对照、易错点和练习参考答案。
- [线性回归互动习题](./3.1%20线性回归/线性回归互动习题.md)：进入 StudyMe.ai，完成 30 道带代码理解和逐题解析的互动练习。

### 这一节要掌握什么

完成 3.1 后，应能够解释并手动写出下面的训练闭环：

```text
读取一个批次 X、y
    ↓
使用 y_hat = Xw + b 得到预测值
    ↓
使用平方损失衡量 y_hat 和 y 的差距
    ↓
调用 backward() 计算 w、b 的梯度
    ↓
按照 theta = theta - learning_rate × gradient 更新参数
    ↓
清空梯度并处理下一个批次
```

特别要注意以下关系：

| 主题 | 需要理解的内容 |
| --- | --- |
| 张量形状 | `X` 通常是 `(batch_size, num_features)`，`w` 通常是 `(num_features, 1)` |
| 计算图 | `requires_grad=True` 让 PyTorch 记录与参数有关的运算 |
| 反向传播 | `loss.backward()` 根据计算图计算梯度 |
| 梯度清零 | 梯度默认累加，需要使用 `zero_grad()` 或 `grad.zero_()` |
| 参数更新 | 梯度下降沿梯度反方向移动 |
| 推理阶段 | 不需要训练时可以使用 `torch.no_grad()` |

## 环境准备

项目未提供固定的 `requirements.txt`，建议使用 Python 3.9+ 创建虚拟环境，并按实验需要安装依赖：

## 环境准备

项目未提供固定的 `requirements.txt`，建议使用 Python 3.9+ 创建虚拟环境，并按需安装依赖：

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install torch torchvision numpy pandas matplotlib pillow jieba jupyter
```

如果只阅读基础 Notebook，可以先安装：

```bash
pip install torch numpy jupyter
```

如果运行图像实验，再安装 `torchvision`、`matplotlib` 和 `pillow`；运行中文情感分类实验还需要 `pandas` 和 `jieba`。

是否使用 CUDA 取决于本机显卡、驱动以及 PyTorch 安装版本。运行下面的脚本可以查看 PyTorch、CUDA 和 cuDNN 状态：

```bash
python "测试cuda版本.py"
```

没有 GPU 时，应使用带 CPU 判断的代码，或将设备明确设置为 `cpu`。

## 快速开始

在仓库根目录执行：

```bash
# 打开 Notebook
jupyter notebook

# 运行一个基础示例
python "2.1数据操作/张量生成.py"

# 检查 PyTorch、CUDA 和 cuDNN
python "测试cuda版本.py"
```

也可以直接打开各章节目录中的 `.ipynb` 文件，按照笔记和训练题逐步练习。建议从仓库根目录启动 Jupyter，这样脚本中的相对路径更容易正常工作。

### 推荐的 3.1 上手顺序

```text
1. 阅读线性回归完整整理.md 的概念和公式
2. 从上到下运行线性回归从零开始.ipynb
3. 观察 features、labels、w、b 和 y_hat 的 shape
4. 对照 sgd、backward、zero_grad 理解一次参数更新
5. 阅读线性回归简洁实现.ipynb
6. 完成 StudyMe.ai 互动习题并查看解析
```

## 运行实验

```bash
# MNIST 卷积分类
python "convolution/手写数字识别器.py"

# 中文文本情感分类
python "EmotionClassification/文本情感分类器.py"

# 蚂蚁/蜜蜂迁移学习
python "TransferingLearning/迁移学习.py"

# 图像风格迁移
python "picturestyle/风格迁移.py"
```

### 3.1 互动练习

互动课程地址：[StudyMe.ai：3.1 线性回归代码理解、训练实践与逐题解析（30题）](https://studyme.ai/57n7n/)

课程包含选择题、计算题、拖拽匹配题、下拉选择题和公式计算题，重点练习：

- `reshape`、切片和矩阵乘法；
- `requires_grad`、`backward`、`grad` 和 `zero_grad`；
- `torch.no_grad()`、手写 SGD 和学习率；
- `DataLoader`、`nn.Linear`、`MSELoss` 和优化器；
- 广播错误、设备不一致和训练循环顺序。

运行综合实验前请注意：

1. 部分脚本会下载或读取数据，并可能生成模型文件、图像或训练输出。
2. 迁移学习脚本默认从 `TransferingLearning/data/` 读取数据；数据集目录需要包含 `train` 和 `val`，每个目录下再按类别放置图片。
3. `read_data.py` 默认读取根目录下的 `dataset/train`，请从仓库根目录运行，或按本地路径修改脚本。
4. 个别示例包含 CUDA 设备设置。没有 GPU 时，优先使用已包含 CPU 判断的脚本；需要时可将设备改为 `cpu`。

5. `3.1 线性回归/线性回归从零开始.ipynb` 中的部分变量依赖前面单元格的执行结果。若单独运行训练单元格，可能出现变量未定义或沿用旧内核变量的情况，建议从上到下运行。
6. `convolution` 中的 MNIST 数据可能在首次运行时下载；`TransferingLearning` 和 `read_data.py` 需要对应的数据目录结构。
7. Notebook 运行后可能产生 `.ipynb_checkpoints`、模型文件或图像输出。提交代码前请确认是否需要保留这些生成文件。

## 目录结构

```text
Deeplearning/
├── 2.1数据操作/           # 张量与数据操作
├── 2.2数据预处理/         # 数据预处理
├── 2.3线性代数/           # 线性代数基础
├── 2.4矩阵计算/           # 矩阵计算与求导
├── 2.5自动求导/           # 自动求导
├── 2.6 概率/              # 概率
├── 3.1 线性回归/          # 线性回归与优化
│   ├── 线性回归从零开始.ipynb
│   ├── 线性回归简洁实现.ipynb
│   ├── 线性回归完整整理.md
│   └── 线性回归互动习题.md
├── convolution/           # MNIST 卷积分类
├── EmotionClassification/ # 文本情感分类
├── TransferingLearning/   # 图像迁移学习
├── picturestyle/          # 风格迁移
├── data/                  # 示例数据
└── dataset/               # 蚂蚁/蜜蜂数据集
```

## 说明

这是一个以学习和实验为目的的个人仓库。示例代码和笔记会持续整理，运行结果可能因 Python、PyTorch、硬件、随机种子和数据集版本不同而有所差异。

仓库中的示例优先服务于概念理解，代码风格和工程完整性不一定等同于生产项目。阅读代码时，建议同时关注：输入输出形状、数据类型、设备位置、梯度状态和参数更新顺序。
