# Deeplearning

一个用于整理深度学习学习笔记、PyTorch 基础代码和小型实验的仓库。内容按照“基础知识 → 模型训练 → 综合实验”逐步展开，适合配合 Jupyter Notebook 阅读和练习。

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
| `3.1 线性回归` | 线性回归与优化方法基础 |

### 综合实验

- `convolution`：使用卷积神经网络进行 MNIST 手写数字识别。
- `EmotionClassification`：基于中文分词和词袋模型的文本情感分类。
- `TransferingLearning`：使用预训练 ResNet 进行蚂蚁/蜜蜂图像分类的迁移学习实验。
- `picturestyle`：图像风格迁移实验。

### 数据与脚本

- `data/house_tiny.csv`：数据预处理示例使用的小型 CSV 数据。
- `dataset/`：蚂蚁/蜜蜂图像数据，按 `train`、`val` 和类别目录组织。
- 根目录下的 `BasicFunc.py`、`autograd.py`、`read_data.py`、`维度变换.py`、`测试cuda版本.py` 等文件：独立的基础示例和环境检查脚本。

## 环境准备

项目未提供固定的 `requirements.txt`，建议使用 Python 3.9+ 创建虚拟环境，并按需安装依赖：

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install torch torchvision numpy pandas matplotlib pillow jieba jupyter
```

如果只阅读基础 Notebook，可以先安装 `torch`、`numpy` 和 `jupyter`。是否使用 CUDA 取决于本机显卡、驱动以及 PyTorch 安装版本；可运行 `测试cuda版本.py` 检查环境。

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

也可以直接打开各章节目录中的 `.ipynb` 文件，按照笔记和训练题逐步练习。

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

运行综合实验前请注意：

1. 部分脚本会下载或读取数据，并可能生成模型文件、图像或训练输出。
2. 迁移学习脚本默认从 `TransferingLearning/data/` 读取数据；数据集目录需要包含 `train` 和 `val`，每个目录下再按类别放置图片。
3. `read_data.py` 默认读取根目录下的 `dataset/train`，请从仓库根目录运行，或按本地路径修改脚本。
4. 个别示例包含 CUDA 设备设置。没有 GPU 时，优先使用已包含 CPU 判断的脚本；需要时可将设备改为 `cpu`。

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
├── convolution/           # MNIST 卷积分类
├── EmotionClassification/ # 文本情感分类
├── TransferingLearning/   # 图像迁移学习
├── picturestyle/          # 风格迁移
├── data/                  # 示例数据
└── dataset/               # 蚂蚁/蜜蜂数据集
```

## 说明

这是一个以学习和实验为目的的个人仓库。示例代码和笔记会持续整理，运行结果可能因 Python、PyTorch、硬件和数据集版本不同而有所差异。
